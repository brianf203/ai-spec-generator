"""Schema validation."""
def validate_schema(data, schema): return all(k in data for k in schema)
def get_schema_keys(schema): return list(schema.keys()) if isinstance(schema, dict) else []
def validate_schema_types(data, schema): return all(isinstance(data.get(k), schema[k]) for k in schema if k in data)
def validate_schema_required(data, required): return all(k in data for k in required)
def validate_schema_optional(data, optional): return True
def schema_merge(a, b): return {**a, **b}
def schema_defaults(schema, defaults): return {**defaults, **schema}
def schema_pick(schema, keys): return {k: schema[k] for k in keys if k in schema}
def schema_omit(schema, keys): return {k: v for k, v in schema.items() if k not in keys}
def schema_validate_nested(data, schema): return all(validate_schema(data.get(k, {}), schema.get(k, {})) for k in schema if k in data)
def schema_keys_intersection(a, b): return [k for k in a if k in b]
def schema_keys_union(a, b): return list(dict.fromkeys(list(a.keys()) + list(b.keys())))
def schema_diff(a, b): return {k: a[k] for k in a if k not in b or a[k] != b[k]}
def schema_validate_format(data, formats): return True
def schema_to_json_schema(schema): return {"type": "object", "properties": schema}
def schema_from_json_schema(js): return js.get("properties", {})