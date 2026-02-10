import requests
with open('uploads/test_fire_prediction.csv','rb') as f:
    r = requests.post('http://127.0.0.1:5000/uploads', files={'file': f}, allow_redirects=True)
text = r.text
print('contains_prediction_result:', 'Prediction Result' in text or 'High Chance' in text or 'Low Chance' in text)
start = text.find('<h1>Model Results</h1>')
if start!=-1:
    print('\n--- HTML snippet ---\n')
    print(text[start:start+800])
else:
    print('\nModel Results header not found. Response length:', len(text))
