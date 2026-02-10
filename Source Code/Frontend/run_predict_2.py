from app import make_prediction
r = make_prediction('uploads/test_fire_prediction_2.csv')
print('Result:', r[0])
print('Probability:', r[1])
print('Error:', r[2])
