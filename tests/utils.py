def assert_json_structure(json, struct: type | list | dict):
    if isinstance(struct, list):
        assert isinstance(json, list)
        sub_struct = struct[0]
        for item in json:
            assert_json_structure(item, sub_struct)
    elif isinstance(struct, dict):
        assert isinstance(json, dict)
        for key, sub_struct in struct.items():
            assert_json_structure(json.get(key, None), sub_struct)
    else:
        assert isinstance(json, struct)
