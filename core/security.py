"""
NAOMI Agent - Security Module
Handles: prompt injection defense, input sanitization, sensitive operation
confirmation, audit logging, and periodic security scanning.
"""
import os
import re
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("naomi.security")

AUDIT_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "audit.log")
SECURITY_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "security.log")


# === 1. Prompt Injection Defense ===

def sanitize_external_content(content: str, source: str = "unknown") -> str:
    """Wrap external content with isolation markers.
    Prevents prompt injection from web pages, emails, API responses, etc.
    """
    if not content:
        return content

    # Truncate extremely long content (context window poisoning)
    max_len = 5000
    if len(content) > max_len:
        content = content[:max_len] + f"\n[...truncated from {len(content)} chars]"

    # Strip known injection patterns
    injection_patterns = [
        r'(?i)ignore\s+(previous|all|above)\s+(instructions?|prompts?)',
        r'(?i)you\s+are\s+now\s+',
        r'(?i)new\s+instructions?:',
        r'(?i)system\s*prompt\s*:',
        r'(?i)forget\s+(everything|all|previous)',
        r'(?i)\[SYSTEM\]',
        r'(?i)ADMIN\s*OVERRIDE',
        r'(?i)jailbreak',
    ]

    flagged = False
    for pattern in injection_patterns:
        if re.search(pattern, content):
            flagged = True
            content = re.sub(pattern, '[BLOCKED: injection attempt]', content)

    if flagged:
        logger.warning(f"Prompt injection detected in content from {source}")
        log_security_event("prompt_injection_blocked", {
            "source": source,
            "content_preview": content[:200],
        })

    return f"<external_content source=\"{source}\">\n{content}\n</external_content>"


# === 3. Sensitive Operation Confirmation ===

# Operations that require Master confirmation via Telegram
SENSITIVE_OPERATIONS = {
    "rm_rf": {
        "patterns": [r'rm\s+-rf\s+/', r'rm\s+-rf\s+~', r'rm\s+-rf\s+\*'],
        "description": "刪除大量檔案",
    },
    "git_force": {
        "patterns": [r'git\s+push\s+.*--force', r'git\s+push\s+-f\s', r'git\s+reset\s+--hard'],
        "description": "Git 強制操作",
    },
    "system_modify": {
        "patterns": [r'sudo\s+rm', r'sudo\s+mv\s+/', r'chmod\s+-R\s+777',
                     r'chown\s+-R', r'launchctl\s+unload'],
        "description": "系統級修改",
    },
    "network_expose": {
        "patterns": [r'ngrok', r'0\.0\.0\.0', r'--host\s+0\.0\.0\.0'],
        "description": "對外暴露網路服務",
    },
}


def check_sensitive_command(command: str) -> Optional[Dict]:
    """Check if a command is sensitive and needs confirmation.
    Returns None if safe, or a dict with details if sensitive.
    """
    for op_name, op_info in SENSITIVE_OPERATIONS.items():
        for pattern in op_info["patterns"]:
            if re.search(pattern, command):
                return {
                    "operation": op_name,
                    "description": op_info["description"],
                    "command": command[:200],
                    "needs_confirmation": True,
                }
    return None


# === 4. Input Sanitization ===

def sanitize_telegram_input(text: str) -> str:
    """Sanitize Telegram input to prevent injection attacks."""
    if not text:
        return text

    # Remove null bytes
    text = text.replace('\x00', '')

    # Limit length (prevent context flooding)
    max_input = 4000
    if len(text) > max_input:
        text = text[:max_input] + "..."

    return text


def validate_master_id(user_id: int, master_id: int) -> bool:
    """Validate Telegram user is the authorized master."""
    return user_id == master_id


# === 5. Secret Management ===

def scan_for_leaked_secrets(project_dir: str) -> List[Dict]:
    """Scan project files for accidentally committed secrets."""
    leaks = []
    secret_patterns = [
        (r'sk-[a-zA-Z0-9]{20,}', "API key (sk-...)"),
        (r'[0-9]+:[A-Za-z0-9_-]{35}', "Telegram bot token"),
        (r'ghp_[A-Za-z0-9]{36}', "GitHub PAT"),
        (r'AKIA[0-9A-Z]{16}', "AWS Access Key"),
        (r'-----BEGIN (RSA |EC )?PRIVATE KEY-----', "Private key"),
        (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
    ]

    skip_dirs = {'__pycache__', '.git', 'node_modules', 'data', '.env'}
    skip_extensions = {'.pyc', '.db', '.log', '.png', '.jpg', '.bak'}

    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if any(f.endswith(ext) for ext in skip_extensions):
                continue
            if f == '.env':
                continue  # .env is expected to have secrets

            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'r', errors='ignore') as fh:
                    content = fh.read()
                for pattern, desc in secret_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        rel_path = os.path.relpath(filepath, project_dir)
                        leaks.append({
                            "file": rel_path,
                            "type": desc,
                            "count": len(matches),
                        })
            except Exception:
                pass

    return leaks


# === 6. Audit Logging ===

def audit_log(action: str, tool: str, params: str, result_summary: str = "",
              success: bool = True):
    """Log a tool invocation to the audit log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "tool": tool,
        "params": params[:200],
        "result": result_summary[:200],
        "success": success,
    }

    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
        with open(AUDIT_LOG_FILE, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"Audit log write failed: {e}")


def get_recent_audit(limit: int = 50) -> List[Dict]:
    """Get recent audit log entries."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    try:
        with open(AUDIT_LOG_FILE, 'r') as f:
            lines = f.readlines()
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
        return entries
    except Exception:
        return []


def rotate_audit_log(max_lines: int = 5000):
    """Rotate audit log if it gets too large."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return
    try:
        with open(AUDIT_LOG_FILE, 'r') as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            with open(AUDIT_LOG_FILE, 'w') as f:
                f.writelines(lines[-max_lines:])
            logger.info(f"Audit log rotated: {len(lines)} → {max_lines}")
    except Exception:
        pass


# === 7. Security Event Logging ===

def log_security_event(event_type: str, details: Dict):
    """Log a security-relevant event."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **details,
    }
    try:
        os.makedirs(os.path.dirname(SECURITY_LOG_FILE), exist_ok=True)
        with open(SECURITY_LOG_FILE, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    logger.warning(f"Security event: {event_type} — {json.dumps(details)[:200]}")


# === Full Security Scan ===

def run_security_scan(project_dir: str) -> Dict:
    """Run a comprehensive security scan."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "issues": [],
    }

    # Check for leaked secrets
    leaks = scan_for_leaked_secrets(project_dir)
    for leak in leaks:
        results["issues"].append({
            "severity": "CRITICAL",
            "type": "secret_leak",
            **leak,
        })

    # Check .env permissions
    env_file = os.path.join(project_dir, ".env")
    if os.path.exists(env_file):
        import stat
        mode = os.stat(env_file).st_mode
        if mode & stat.S_IROTH:  # World-readable
            results["issues"].append({
                "severity": "HIGH",
                "type": "env_permissions",
                "detail": ".env is world-readable",
                "fix": f"chmod 600 {env_file}",
            })

    # Check .gitignore
    gitignore = os.path.join(project_dir, ".gitignore")
    if os.path.exists(gitignore):
        with open(gitignore, 'r') as f:
            gi_content = f.read()
        required = [".env", "data/", "*.db"]
        for req in required:
            if req not in gi_content:
                results["issues"].append({
                    "severity": "MEDIUM",
                    "type": "gitignore_missing",
                    "detail": f".gitignore missing: {req}",
                })

    # Check dashboard token
    token_file = os.path.join(project_dir, "data", "dashboard_token.txt")
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token = f.read().strip()
        if len(token) < 20:
            results["issues"].append({
                "severity": "HIGH",
                "type": "weak_dashboard_token",
                "detail": "Dashboard token is too short",
            })

    # Check audit log exists
    if not os.path.exists(AUDIT_LOG_FILE):
        results["issues"].append({
            "severity": "LOW",
            "type": "no_audit_log",
            "detail": "No audit log found — tool actions not being tracked",
        })

    results["total_issues"] = len(results["issues"])
    results["critical"] = sum(1 for i in results["issues"] if i["severity"] == "CRITICAL")
    results["high"] = sum(1 for i in results["issues"] if i["severity"] == "HIGH")

    return results
