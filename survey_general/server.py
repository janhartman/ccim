import random

from flask import Flask, render_template, request

app = Flask(__name__)

# Each pair has:
# - one image from our algorithm with all steps
# - one image without MDS (2d positions are random), but still run through our algorithm
pairs = []
for id_ in [2, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15]:
    if random.randrange(2):
        im1 = f'{id_}_default.png'
        im2 = f'{id_}_no_mds.png'
    else:
        im1 = f'{id_}_no_mds.png'
        im2 = f'{id_}_default.png'

    pairs.append({
        'id': str(id_),
        'image1': im1,
        'image2': im2
    })


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html', pairs=pairs)


responses = []
response_file = None

try:
    response_file = open('responses.txt', 'a')
except Exception as e:
    print('Error when trying to open file for writing: ', e)


@app.route('/response', methods=['POST'])
def response():
    winners = {}
    for pair in pairs:
        winner = request.form.get(pair['id'], None)
        if winner:
            winners[pair['id']] = winner
        else:
            raise ValueError('Pair ' + pair['id'] + ' has no winner')

    winners['name'] = request.form.get('name', None)
    print(winners)
    responses.append(winners)
    response_file.write(str(winners) + ', ')
    response_file.flush()

    return render_template('submitted.html')

