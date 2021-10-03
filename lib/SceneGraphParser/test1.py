import sng_parser
import urllib.request
import urllib.parse
import json

def translation(content):

    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'
    data = {'i': content, 'doctype': 'json'}  # 定义一个字典，将data数据存入

    data = urllib.parse.urlencode(data).encode('utf-8')
    response = urllib.request.urlopen(url, data)
    html = response.read().decode('utf-8')
    target = json.loads(html)
    print("翻译的结果为 : %s" % (target['translateResult'][0][0]['tgt']))


sentence = 'A black Honda motorcycle parked in front of a garage'
translation(sentence)
print(sng_parser.parse(sentence)["relations"])
