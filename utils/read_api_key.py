def read_api_keys(api_file, num = 20):
    apis = []
    data = open(api_file, 'r').readlines()
    for example in data:
        items = example.split('----')
        for item in items:
            if item.startswith('sk-'):
                apis.append(item)
        if len(apis) >= num:
            break
    print(f"import {len(apis)} api keys")
    return apis