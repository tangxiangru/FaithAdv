import json

def convert(content: str):
    """
    convert the content of json file in string format into json
    """
    content_json = json.loads(content)
    return content_json

content = "{\n\t\"subject_replaced\": \"Report rejects theory that animal-made seismic activity is to blame.\",\n\t\"predicate_replaced\": \"Report accepts theory that man-made seismic activity is to blame.\",\n\t\"name_replaced\": \"no_name\"\n}"
# content = '{"subject_replaced": "Report rejects theory that animal-made seismic activity is to blame.","predicate_replaced": "Report accepts theory that man-made seismic activity is to blame.","name_replaced": "no_name"}'
# content = '{"a": "11 11 11_!", "b": "22"}'
# content = "{'subject_replaced': 'Report rejects theory that animal-made seismic activity is to blame.','predicate_replaced': 'Report accepts theory that man-made seismic activity is to blame.','name_replaced': 'no_name'}"
content_json = convert(content)
for item in content_json:
    print(item, ":", content_json[item])
