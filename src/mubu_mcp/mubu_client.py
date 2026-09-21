"""Mubu MCP — API client with pluggable cache integration.

Ported from mubu-integration (liuboacean/mubu-integration) and adapted
to use the CacheBackend abstraction for token / user / document caching.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from mubu_mcp.cache.base import CacheBackend
from mubu_mcp.mubu_config import (
    BASE_URL,
    DEFAULT_HEADERS,
    ENDPOINTS,
    MAX_NETWORK_RETRIES,
    MAX_SEARCH_DEPTH,
    MAX_SEARCH_LIMIT,
    MAX_SEARCH_REQUESTS,
    NETWORK_BACKOFF,
    REQUEST_TIMEOUT,
    MubuError,
    logger,
)
from mubu_mcp.mubu_convert import (
    export_markdown,
    gen_member_id,
    gen_node_id,
    normalize_node,
    safe_filename,
)


class MubuClient:
    """Mubu API client backed by a pluggable cache."""

    def __init__(
        self,
        phone: Optional[str] = None,
        password: Optional[str] = None,
        cache: Optional[CacheBackend] = None,
        member_id: Optional[str] = None,
    ) -> None:
        self._load_env_file()

        self.phone = phone or os.getenv("MUBU_PHONE")
        self.password = password or os.getenv("MUBU_PASSWORD")

        self.token: Optional[str] = None
        self.user_id: Optional[str] = None
        self.username: Optional[str] = None
        self.member_id: Optional[str] = None
        self.expires_at: float = 0.0

        self._cache = cache
        self._session = requests.Session()
        self._client_unique_id = str(uuid.uuid4())
        self._session_id = str(uuid.uuid4())

        # Restore cached token
        self._load_token()

        # member_id: env/arg wins; otherwise generate a fresh random one
        # (the web client issues a random 16-digit member id per editing
        # session; the plain user id is NOT a valid member id and its use
        # makes the server accept saves without persisting content).
        self.member_id = member_id or os.getenv("MUBU_MEMBER_ID") or gen_member_id()

    # ------------------------------------------------------------------
    # Env file loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_env_file() -> None:
        # 候选路径：项目级 .env 优先，再兼容 ~/.workbuddy/.env.mubu
        project_env = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".env",
        )
        env_paths = [project_env, os.path.expanduser("~/.workbuddy/.env.mubu")]
        for env_path in env_paths:
            if not os.path.isfile(env_path):
                continue
            try:
                for line in open(env_path, encoding="utf-8").read().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip().strip("\"'")
                    if key in ("MUBU_PHONE", "MUBU_PASSWORD", "MUBU_MEMBER_ID") and not os.getenv(key):
                        os.environ[key] = value
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Token persistence (via cache or file fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _jwt_expiry(token: str) -> Optional[float]:
        """Decode the ``exp`` claim from a JWT (seconds since epoch).

        Returns ``None`` when the token is not a decodable JWT.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            payload_b64 = parts[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            exp = payload.get("exp")
            return float(exp) if exp else None
        except Exception:
            return None

    def _load_token(self) -> bool:
        # Try cache first
        if self._cache:
            data = self._cache.load_token()
            if data and self._token_still_valid(data):
                self.token = data["token"]
                self.user_id = data.get("user_id")
                self.username = data.get("username")
                self.member_id = data.get("member_id")
                self.expires_at = data["expires_at"]
                return True

        # File fallback
        token_file = os.path.expanduser("~/.mubu_token")
        if os.path.isfile(token_file):
            try:
                data = json.loads(open(token_file, encoding="utf-8").read())
                if self._token_still_valid(data):
                    self.token = data.get("token")
                    self.user_id = data.get("user_id")
                    self.username = data.get("username")
                    self.member_id = data.get("member_id")
                    self.expires_at = data["expires_at"]
                    # Promote to cache
                    if self._cache:
                        self._cache.store_token(
                            self.token,  # type: ignore[arg-type]
                            self.user_id or "",  # type: ignore[arg-type]
                            self.username or "",  # type: ignore[arg-type]
                            self.member_id,
                        )
                    return True
            except Exception:
                pass
        return False

    def _token_still_valid(self, data: Dict[str, Any]) -> bool:
        """Token record is usable when its local expiry is in the future, or
        when the JWT itself is still valid (renew the local expiry to the JWT
        exp to avoid a needless re-login)."""
        exp = data.get("expires_at") or 0
        try:
            exp = float(exp)
        except (TypeError, ValueError):
            exp = 0
        if time.time() < exp:
            return True
        jwt_exp = self._jwt_expiry(str(data.get("token") or ""))
        if jwt_exp and time.time() < jwt_exp:
            data["expires_at"] = jwt_exp
            return True
        return False

    def _save_token(self) -> None:
        jwt_exp = self._jwt_expiry(self.token or "")
        self.expires_at = jwt_exp or (time.time() + 7200)
        data = {
            "token": self.token,
            "user_id": self.user_id,
            "username": self.username,
            "member_id": self.member_id,
            "expires_at": self.expires_at,
        }
        if self._cache:
            self._cache.store_token(
                self.token or "",  # type: ignore[arg-type]
                self.user_id or "",  # type: ignore[arg-type]
                self.username or "",  # type: ignore[arg-type]
                self.member_id,
            )
        # Also write file for backward-compat
        token_file = os.path.expanduser("~/.mubu_token")
        try:
            os.makedirs(os.path.dirname(token_file), exist_ok=True)
            tmp = token_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.rename(tmp, token_file)
            os.chmod(token_file, 0o600)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get_headers(self) -> Dict[str, str]:
        headers = DEFAULT_HEADERS.copy()
        if self.token:
            headers["Jwt-Token"] = self.token
        headers["data-unique-id"] = self._client_unique_id
        headers["x-session-id"] = f"{self._session_id}:{int(time.time())}"
        headers["x-reg-entrance"] = "https://mubu.com/app"
        headers["x-request-id"] = str(uuid.uuid4())
        return headers

    def ensure_valid_token(self) -> None:
        if not self.token or time.time() > self.expires_at - 360:
            self.login()

    def _is_auth_error(self, result: Dict[str, Any], response: requests.Response) -> bool:
        if response.status_code == 401:
            return True
        code = result.get("code")
        if code is not None and code != 0:
            msg = str(result.get("msg", "")).lower()
            for kw in ("登录", "未登录", "重新登录", "登录失效", "unauthorized"):
                if kw in msg:
                    return True
        return False

    def _http_request(self, method: str, url: str, headers: Dict[str, str], **kwargs: Any) -> requests.Response:
        last_err: Optional[Exception] = None
        for attempt in range(MAX_NETWORK_RETRIES + 1):
            try:
                response = self._session.request(method, url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)
            except requests.exceptions.RequestException as e:
                last_err = e
                if attempt < MAX_NETWORK_RETRIES:
                    time.sleep(NETWORK_BACKOFF[min(attempt, len(NETWORK_BACKOFF) - 1)])
                    continue
                raise MubuError(f"Network error after {MAX_NETWORK_RETRIES} retries: {e}")

            if response.status_code == 429:
                if attempt < MAX_NETWORK_RETRIES:
                    retry_after = response.headers.get("Retry-After")
                    wait = min(int(retry_after), 30) if retry_after and str(retry_after).isdigit() else NETWORK_BACKOFF[min(attempt, len(NETWORK_BACKOFF) - 1)]
                    time.sleep(wait)
                    continue
                raise MubuError("Rate limited (HTTP 429)", status_code=429)

            if response.status_code >= 500:
                if attempt < MAX_NETWORK_RETRIES:
                    time.sleep(NETWORK_BACKOFF[min(attempt, len(NETWORK_BACKOFF) - 1)])
                    continue
                raise MubuError(f"Server unavailable (HTTP {response.status_code})", status_code=response.status_code)

            return response

        raise MubuError(f"Request failed: {last_err}")

    def _request(self, method: str, endpoint: str, *, max_retries: int = 1, auth: bool = True, **kwargs: Any) -> Dict:
        if auth:
            self.ensure_valid_token()
        url = f"{BASE_URL}{endpoint}"
        headers = self._get_headers()
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        response = self._http_request(method, url, headers, **kwargs)

        try:
            result = response.json()
        except ValueError:
            raise MubuError(f"Non-JSON response (status={response.status_code})", status_code=response.status_code, body=response.text[:300])

        if self._is_auth_error(result, response):
            if max_retries > 0:
                self.login()
                return self._request(method, endpoint, max_retries=max_retries - 1, auth=auth, **kwargs)
            raise MubuError(f"Auth failed: {result.get('msg', 'unknown')}", status_code=response.status_code, body=result)

        if result.get("code") != 0:
            if response.status_code == 403:
                raise MubuError(f"Forbidden: {result.get('msg', 'unknown')}", status_code=403, body=result)
            raise MubuError(f"API error: {result.get('msg', 'unknown')}", status_code=response.status_code, body=result)

        return result.get("data", {})

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self) -> Dict:
        if not self.phone or not self.password:
            raise MubuError("Set MUBU_PHONE and MUBU_PASSWORD env vars or pass them to the constructor.")

        try:
            data = self._request(*ENDPOINTS["login"], auth=False, max_retries=0, json={
                "phone": self.phone,
                "password": self.password,
                "callbackType": 0,
            })
        except MubuError as exc:
            if "frequency" in str(exc).lower() or "login" in str(exc).lower():
                raise MubuError(
                    "Login rate limited by Mubu ('Login Frequency'). "
                    "A valid cached token exists but the login endpoint is "
                    "throttled; wait a few minutes before retrying.",
                    status_code=getattr(exc, "status_code", None),
                    body=getattr(exc, "body", None),
                ) from exc
            raise
        self.token = data["token"]
        self.user_id = data["id"]
        self.username = data["name"]
        # memberId is not exposed by the login API; keep whatever session
        # member id was already generated (or generate one).
        if not self.member_id:
            self.member_id = gen_member_id()
        self._save_token()
        return {"token": self.token, "user_id": self.user_id, "username": self.username}

    def ensure_login(self) -> None:
        if not self.token:
            self.login()

    # ------------------------------------------------------------------
    # Document / folder operations
    # ------------------------------------------------------------------

    def get_list(self, folder_id: str = "0") -> Dict:
        return self._request(*ENDPOINTS["list"], json={"folderId": folder_id})

    def create_folder(self, name: str, parent_id: str = "0") -> str:
        data = self._request(*ENDPOINTS["create_folder"], json={"folderId": parent_id, "name": name})
        return data.get("folder", {}).get("id", "")

    def create_doc(self, name: str, folder_id: str = "0", content: str = "") -> str:
        data = self._request(*ENDPOINTS["create_doc"], json={"folderId": folder_id, "name": name, "content": content})
        doc_id = data.get("id", "") or (data.get("doc", {}) or {}).get("id", "")
        if self._cache and doc_id:
            self._cache.invalidate_doc(doc_id)
        return doc_id

    def get_doc(self, doc_id: str) -> Dict:
        # Check cache first
        if self._cache:
            cached = self._cache.load_doc(doc_id)
            if cached:
                return cached

        self.ensure_login()
        data = self._request(*ENDPOINTS["get_doc"], json={
            "docId": doc_id, "password": "", "isFromDocDir": True,
        })
        try:
            definition = json.loads(data["definition"])
        except (KeyError, TypeError, ValueError) as e:
            raise MubuError(f"Failed to parse doc definition (doc_id={doc_id}): {e}") from e

        result = {"name": data.get("name"), "nodes": definition.get("nodes", [])}

        # Cache the result
        if self._cache:
            self._cache.store_doc(doc_id, result)

        return result

    def get_doc_meta(self, doc_id: str) -> Dict:
        """Fetch document metadata (parent folder, name, base version) without
        node caching. Used by save flows that may need to recreate a doc."""
        self.ensure_login()
        data = self._request(*ENDPOINTS["get_doc"], json={
            "docId": doc_id, "password": "", "isFromDocDir": True,
        })
        directory = data.get("directory") or []
        folder_id = "0"
        if directory:
            folder_id = directory[-1].get("id") or "0"
        return {
            "folder_id": folder_id,
            "name": data.get("name") or "",
            "base_version": data.get("baseVersion") or 0,
            "definition": data.get("definition") or "{}",
        }

    # ------------------------------------------------------------------
    # Changeset event builders (real Mubu colla/events schema)
    # ------------------------------------------------------------------

    def build_create_root_event(self, root_node: Dict) -> Dict:
        """Event that creates the root node of an empty document, carrying a
        full subtree. Verified against the live server."""
        return {
            "name": "create",
            "created": [{"index": 0, "parentId": None, "node": root_node, "path": ["nodes", 0]}],
        }

    def build_create_children_events(self, parent_id: str, children: List[Dict], start_index: int = 0) -> List[Dict]:
        """Events that append child nodes under ``parent_id`` (index/path
        aligned with the real web client)."""
        events = []
        for i, child in enumerate(children):
            idx = start_index + i
            events.append({
                "name": "create",
                "created": [{
                    "index": idx,
                    "parentId": parent_id,
                    "node": child,
                    "path": ["nodes", 0, "children", idx],
                }],
            })
        return events

    def build_update_root_event(self, old_root: Dict, new_text: str) -> Dict:
        """Event that updates the root node's text (children are NOT changed
        by an update event — verified against the live server)."""
        return {
            "name": "update",
            "updated": [{
                "updated": {
                    "id": old_root.get("id", ""),
                    "text": f"<span>{str(new_text or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</span>",
                    "modified": int(time.time() * 1000),
                },
                "original": {
                    "id": old_root.get("id", ""),
                    "text": old_root.get("text", ""),
                    "modified": old_root.get("modified", 0),
                },
                "path": ["nodes", 0],
            }],
        }

    def build_update_event(self, doc_definition: Dict, doc_id: str) -> Dict:
        """(Legacy) Build an update event for the whole document. The server
        only applies the root text; use the explicit builders for real
        structure changes."""
        nodes = doc_definition.get("nodes", []) if isinstance(doc_definition, dict) else []
        normalized_nodes = [normalize_node(n) for n in nodes]
        root = {"id": doc_id, "children": normalized_nodes, "modified": int(time.time() * 1000)}
        return {"name": "update", "updated": [{"updated": root, "original": root}]}

    def save_doc(self, doc_id: str, events: Optional[List[Dict]] = None, version: Optional[int] = None, name: Optional[str] = None) -> None:
        if version is None or events is None:
            raw = self._request(*ENDPOINTS["get_doc"], json={"docId": doc_id, "password": "", "isFromDocDir": True})
            if version is None:
                version = raw.get("baseVersion")
            if events is None:
                definition = json.loads(raw["definition"])
                events = [self.build_update_event(definition, doc_id)]

        if not self.member_id:
            # Random session member id (web client behaviour); the plain user
            # id is not a valid member id.
            self.member_id = gen_member_id()

        payload = {
            "memberId": self.member_id,
            "type": "CHANGE",
            "version": version,
            "documentId": doc_id,
            "events": events,
        }
        headers = {"x-reg-entrance": f"https://mubu.com/app/edit/home/{doc_id}"}
        self._request(*ENDPOINTS["save_doc"], json=payload, headers=headers)

        if self._cache:
            self._cache.invalidate_doc(doc_id)

        if name:
            self.rename_doc(doc_id, name)

    def delete_folder(self, folder_id: str) -> None:
        self._request(*ENDPOINTS["delete_folder"], json={"id": folder_id})

    def delete_doc(self, doc_id: str) -> None:
        self._request(*ENDPOINTS["delete_doc"], json={"id": doc_id})
        if self._cache:
            self._cache.invalidate_doc(doc_id)

    def move(self, item_id: str, target_folder_id: str, item_type: str = "doc") -> None:
        self._request(*ENDPOINTS["move"], json={
            "dst": None,
            "src": [{"type": item_type, "id": item_id}],
            "folderId": target_folder_id,
        })

    def rename_doc(self, doc_id: str, new_name: str) -> None:
        self._request(*ENDPOINTS["rename_doc"], json={"documentId": doc_id, "name": new_name})
        if self._cache:
            self._cache.invalidate_doc(doc_id)

    def rename_folder(self, folder_id: str, new_name: str) -> None:
        self._request("POST", "/list/rename_folder", json={
            "id": folder_id, "name": new_name, "folderId": folder_id,
        })

    # ------------------------------------------------------------------
    # Folder path helpers (resolve / auto-create by slash-separated path)
    # ------------------------------------------------------------------

    def find_folder_by_path(self, path: str, root_folder_id: str = "0") -> Optional[str]:
        """Resolve a slash-separated folder path to a folder ID.

        Path is relative to ``root_folder_id`` (default "0" = root), e.g.
        ``"工作/项目A/子目录"``. Returns the final folder ID, or ``None``
        if any segment does not exist.
        """
        segments = [s for s in (path or "").strip().strip("/").split("/") if s]
        current = root_folder_id
        for seg in segments:
            data = self.get_list(current)
            folders = data.get("folders", []) or []
            found = next((f.get("id") for f in folders if f.get("name") == seg), None)
            if not found:
                return None
            current = found
        return current

    def ensure_folder_path(self, path: str, root_folder_id: str = "0") -> str:
        """Ensure a slash-separated folder path exists, creating missing
        folders level by level. Returns the final folder ID."""
        segments = [s for s in (path or "").strip().strip("/").split("/") if s]
        current = root_folder_id
        for seg in segments:
            data = self.get_list(current)
            folders = data.get("folders", []) or []
            found = next((f.get("id") for f in folders if f.get("name") == seg), None)
            if not found:
                found = self.create_folder(seg, current)
            current = found
        return current

    def doc_exists_in_folder(self, folder_id: str, doc_id: str) -> bool:
        """Check whether a document already lives in the given folder."""
        data = self.get_list(folder_id)
        docs = data.get("documents") or data.get("docs") or []
        return any(d.get("id") == doc_id for d in docs)

    def find_doc_in_folder(self, folder_id: str, name: str) -> Optional[str]:
        """Find a document by exact name inside a folder. Returns its ID or None."""
        data = self.get_list(folder_id)
        docs = data.get("documents") or data.get("docs") or []
        for d in docs:
            if d.get("name") == name:
                return d.get("id")
        return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        keyword: str,
        root_folder_id: str = "0",
        max_depth: int = MAX_SEARCH_DEPTH,
        limit: int = MAX_SEARCH_LIMIT,
        max_requests: int = MAX_SEARCH_REQUESTS,
        include_content: bool = False,
    ) -> Dict[str, Any]:
        keyword_lower = keyword.lower()
        results: List[Dict[str, Any]] = []
        req_count = 0
        truncated = False
        visited: set = set()

        def walk(folder_id: str, path: str, depth: int) -> None:
            nonlocal req_count, truncated
            if folder_id in visited:
                return
            visited.add(folder_id)
            if truncated or depth > max_depth or req_count >= max_requests:
                if depth > max_depth or req_count >= max_requests:
                    truncated = True
                return
            try:
                data = self.get_list(folder_id)
            except MubuError as e:
                logger.warning("Folder traversal failed for %s: %s", folder_id, e)
                return
            req_count += 1
            if len(results) >= limit:
                truncated = True
                return

            folders = data.get("folders", []) or []
            docs = data.get("documents") or data.get("docs") or []

            for d in docs:
                doc_id = d.get("id")
                name = d.get("name") or ""
                if keyword_lower and keyword_lower in name.lower():
                    results.append({"id": doc_id, "name": name, "type": "doc", "path": path, "matched_in": "name"})
                    if len(results) >= limit:
                        truncated = True
                        return
                    continue
                if include_content and keyword_lower:
                    try:
                        doc = self.get_doc(doc_id)
                    except MubuError:
                        continue
                    if _keyword_in_nodes(doc.get("nodes", []), keyword_lower):
                        results.append({"id": doc_id, "name": name, "type": "doc", "path": path, "matched_in": "content"})
                        if len(results) >= limit:
                            truncated = True
                            return

            for f in folders:
                fid = f.get("id")
                fname = f.get("name") or ""
                if keyword_lower and fname.lower() and keyword_lower in fname.lower():
                    results.append({"id": fid, "name": fname, "type": "folder", "path": path})
                    if len(results) >= limit:
                        truncated = True
                        return
                child_path = f"{path}/{fname}" if path else fname
                walk(fid, child_path, depth + 1)

        walk(root_folder_id, "", 0)
        return {"results": results, "truncated": truncated, "limit": limit, "max_depth": max_depth}

    def export_tree(self, root_folder_id: str = "0", max_depth: int = MAX_SEARCH_DEPTH) -> Dict[str, int]:
        """Export a folder tree as in-memory nested Markdown. Returns {docs, errors}."""
        stats: Dict[str, int] = {"docs": 0, "errors": 0}

        def walk(folder_id: str, depth: int) -> Dict[str, Any]:
            if depth > max_depth:
                return {}
            try:
                data = self.get_list(folder_id)
            except MubuError:
                stats["errors"] += 1
                return {}

            result: Dict[str, Any] = {}
            folders = data.get("folders", []) or []
            docs = data.get("documents") or data.get("docs") or []

            for d in docs:
                doc_id = d.get("id")
                name = (d.get("name") or "untitled").strip()
                try:
                    doc = self.get_doc(doc_id)
                    md = export_markdown(doc)
                    result[f"{safe_filename(name)}.md"] = md
                    stats["docs"] += 1
                except MubuError:
                    stats["errors"] += 1

            for f in folders:
                fname = (f.get("name") or "untitled").strip()
                result[safe_filename(fname)] = walk(f.get("id", ""), depth + 1)

            return result

        tree = walk(root_folder_id, 0)
        return {"tree": tree, **stats}


    # ------------------------------------------------------------------
    # Image upload (TOS direct upload via STS + TOS4 signature)
    # ------------------------------------------------------------------

    def _get_tos_sts(self) -> Dict[str, str]:
        """Fetch temporary TOS credentials (AK/SK/sessionToken) for uploads."""
        data = self._request("GET", "/tos/sts")
        return data["credentials"]

    def _tos_put_object(
        self,
        ak: str,
        sk: str,
        sts_token: str,
        key: str,
        content: bytes,
        content_type: str,
    ) -> None:
        """Upload an object to the mubu-img TOS bucket using TOS4-HMAC-SHA256.

        Endpoint / region / bucket / key layout match the web client:
        PUT https://mubu-img.tos-cn-shanghai.volces.com/{key}
        """
        region = "cn-shanghai"
        endpoint = "tos-cn-shanghai.volces.com"
        bucket = "mubu-img"
        algo = "TOS4-HMAC-SHA256"
        host = f"{bucket}.{endpoint}"
        path = "/" + key
        now = datetime.datetime.now(datetime.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(content).hexdigest()

        headers = {
            "host": host,
            "x-tos-date": amz_date,
            "x-tos-content-sha256": payload_hash,
            "x-tos-security-token": sts_token,
            "content-type": content_type,
        }
        signed_headers = ";".join(sorted(headers.keys()))
        canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in sorted(headers.keys()))
        canonical_request = "\n".join(
            ["PUT", path, "", canonical_headers, signed_headers, payload_hash]
        )
        credential_scope = f"{date_stamp}/{region}/tos/request"
        string_to_sign = "\n".join(
            [algo, amz_date, credential_scope, hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()]
        )

        def _sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        k_date = _sign(sk.encode("utf-8"), date_stamp)
        k_region = _sign(k_date, region)
        k_service = _sign(k_region, "tos")
        k_signing = _sign(k_service, "request")
        signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        headers["Authorization"] = (
            f"{algo} Credential={ak}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        url = f"https://{host}{path}"
        resp = self._session.put(url, data=content, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            raise MubuError(
                f"TOS upload failed (status={resp.status_code}): {resp.text[:300]}",
                status_code=resp.status_code,
            )

    def upload_image_bytes(self, data: bytes, filename: str = "image.png") -> Dict[str, Any]:
        """Upload image bytes to Mubu's image storage.

        Returns ``{"key", "url", "ext", "size"}`` where ``key`` is the
        ``document_image/...`` storage key and ``url`` is the public
        ``https://api2.mubu.com/v3/document_image/...`` URL.
        """
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if not ext:
            ext = "png"
        content_type = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
        }.get(ext, "application/octet-stream")

        creds = self._get_tos_sts()
        key = f"document_image/{self.user_id}_{uuid.uuid4()}.{ext}"
        self._tos_put_object(
            creds["accessKeyId"],
            creds["secretAccessKey"],
            creds["sessionToken"],
            key,
            data,
            content_type,
        )
        short = key.split("/", 1)[1]
        url = f"https://api2.mubu.com/v3/document_image/{short}"

        # Sync to recently-used images (web client behaviour)
        try:
            self._request(
                "POST",
                "/document/sync_recently_used_img",
                json={"imageIdList": [key]},
            )
        except MubuError:
            pass

        return {"key": key, "url": url, "ext": ext, "size": len(data)}

    def upload_image_from_path(self, file_path: str) -> Dict[str, Any]:
        """Upload a local image file to Mubu. Returns ``{key, url, ext, size}``."""
        if not os.path.isfile(file_path):
            raise MubuError(f"Image file not found: {file_path}")
        with open(file_path, "rb") as f:
            data = f.read()
        return self.upload_image_bytes(data, os.path.basename(file_path))

    def upload_image_from_url(self, image_url: str) -> Dict[str, Any]:
        """Download an image from a URL and upload it to Mubu.

        Returns ``{key, url, ext, size}``. The ``url`` is the Mubu-hosted
        public URL (avoids external hotlink breakage).
        """
        resp = requests.get(image_url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            raise MubuError(f"Failed to download image (status={resp.status_code}): {image_url}")
        content_type = resp.headers.get("content-type") or ""
        ext = ""
        if "png" in content_type:
            ext = "png"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = "jpg"
        elif "gif" in content_type:
            ext = "gif"
        elif "webp" in content_type:
            ext = "webp"
        else:
            import posixpath
            from urllib.parse import urlparse

            base = posixpath.basename(urlparse(image_url).path)
            ext = os.path.splitext(base)[1].lower().lstrip(".")
            if ext not in ("png", "jpg", "jpeg", "gif", "webp", "bmp"):
                ext = "png"
        return self.upload_image_bytes(resp.content, f"image.{ext}")

    def get_recent_images(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recently used images (most recent first)."""
        data = self._request("GET", "/document/user_img")
        items = data or []
        out = []
        for item in items[:limit]:
            key = item.get("keyImg") or ""
            short = key.split("/", 1)[1] if key.startswith("document_image/") else key
            out.append({
                "key": key,
                "url": f"https://api2.mubu.com/v3/document_image/{short}" if short else "",
                "create_time": item.get("createTime"),
            })
        return out



def _keyword_in_nodes(nodes: Any, keyword_lower: str) -> bool:
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        text = (node.get("text") or "")
        note = (node.get("note") or "")
        if keyword_lower in (text + " " + note).lower():
            return True
        if _keyword_in_nodes(node.get("children", []), keyword_lower):
            return True
    return False
