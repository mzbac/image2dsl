import base64
import requests

url = "http://0.0.0.0:8080/generate_code"

with open("images.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

response = requests.post(url, json={"image_data": encoded_image,"api_key": "YOUR_API_KEY","prompt": "for online shop for selling shoes"})
generated_code = response.json()['generated_code']

print("Generated code:\n", generated_code)
