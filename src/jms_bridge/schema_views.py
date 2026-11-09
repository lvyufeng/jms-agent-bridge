"""Small advertised schemas; full validators and old flat fs calls stay valid."""

from copy import deepcopy


def strip_titles(schema):
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k != "title"}
    for name in ("properties", "$defs", "definitions", "patternProperties"):
        if name in out:
            out[name] = {k: strip_titles(v) for k, v in out[name].items()}
    for name in ("items", "additionalProperties", "contains", "not", "if", "then", "else"):
        if isinstance(out.get(name), dict):
            out[name] = strip_titles(out[name])
    for name in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if name in out:
            out[name] = [strip_titles(v) for v in out[name]]
    return out


def advertised(name, schema):
    schema = strip_titles(schema)
    if name == "fs":
        keep = (
            "target",
            "action",
            "path",
            "pattern",
            "container",
            "options",
            "request_id",
            "detail",
        )
        schema = {**schema, "properties": {k: schema["properties"][k] for k in keep}}
    return schema


FS_COMMON = {"target", "action", "path", "container", "timeout", "max_bytes"}
FS_OPTIONS = {
    "list": {"offset", "limit", "hidden", "exclude"},
    "tree": {"offset", "limit", "hidden", "exclude", "depth"},
    "stat": set(),
    "checksum": {"max_file_bytes"},
    "upload": {"local_path", "overwrite", "exclude", "request_id"},
    "download": {"local_path", "overwrite", "exclude", "request_id"},
    "read": {"start_line", "line_count", "byte_offset", "tail"},
    "read_many": {"paths", "start_line", "line_count", "tail"},
    "glob": {"pattern", "include", "exclude", "hidden", "respect_ignore", "offset", "limit"},
    "search": {
        "pattern",
        "literal",
        "case_sensitive",
        "include",
        "exclude",
        "hidden",
        "respect_ignore",
        "offset",
        "limit",
        "context_lines",
        "max_file_bytes",
    },
}


def fs_action_schema(full, action):
    if action not in FS_OPTIONS:
        raise ValueError("unknown fs action")
    value = deepcopy(full)
    keys = FS_COMMON | FS_OPTIONS[action]
    if action == "read_many":
        keys.discard("path")
    raw = value["inputSchema"]
    raw["properties"] = {k: v for k, v in raw["properties"].items() if k in keys}
    raw["properties"]["action"] = {"type": "string", "const": action}
    raw["required"] = ["target", "action", "paths" if action == "read_many" else "path"]
    if action in ("search", "glob"):
        raw["required"].append("pattern")
    if action in ("upload", "download"):
        raw["required"].append("local_path")
    value["description"] = (
        "fs action=" + action + ". Put advanced fields in fs.options; "
        "call(jms_fs) uses these fields flat. Existing validation applies."
    )
    return strip_titles(value)


def merge_fs_options(flat, options):
    # Identity and recovery IDs remain explicit; options cannot silently retarget.
    reserved = {
        "target",
        "action",
        "path",
        "pattern",
        "container",
        "request_id",
        "detail",
        "options",
    }
    if not options:
        return flat
    invalid = reserved.intersection(options)
    if invalid:
        raise ValueError("fs.options cannot override: " + ", ".join(sorted(invalid)))
    duplicate = set(flat).intersection(options)
    if duplicate:
        raise ValueError("duplicate fs options: " + ", ".join(sorted(duplicate)))
    return {**flat, **options}
