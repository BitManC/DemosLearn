import urllib, sys
import requests

host = 'http://106443b132614215b3dcdf0b570e310c-cn-beijing.alicloudapi.com'
path = '/predict_by_city'
method = 'GET'
appcode = '203879652'
querys = 'JobfLP2uWjy5rzbBVRDHkTOs9mbLNzah'
bodys = {}
url = host + path

request = urllib.Request(url)
request.add_header('Authorization', 'APPCODE ' + appcode)
response = urllib.urlopen(request)
content = response.read()
if (content):
    print(content)
