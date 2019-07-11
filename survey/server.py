from flask import Flask, render_template, request

app = Flask(__name__)


pairs = [
    {
        'image1': '2_default.png',
        'image2': '2_no_mds.png',
        'id': '2'
    },
    {
        'image1': '4_default.png',
        'image2': '4_no_mds.png',
        'id': '4'
    },
    {
        'image1': '5_default.png',
        'image2': '5_no_mds.png',
        'id': '5'
    },
    {
        'image1': '7_default.png',
        'image2': '7_no_mds.png',
        'id': '7'
    },
    {
        'image1': '9_default.png',
        'image2': '9_no_mds.png',
        'id': '9'
    },
    {
        'image1': '10_default.png',
        'image2': '10_no_mds.png',
        'id': '10'
    },
    {
        'image1': '11_default.png',
        'image2': '11_no_mds.png',
        'id': '11'
    },
    {
        'image1': '12_default.png',
        'image2': '12_no_mds.png',
        'id': '12'
    },
    {
        'image1': '13_default.png',
        'image2': '13_no_mds.png',
        'id': '13'
    },
    {
        'image1': '14_default.png',
        'image2': '14_no_mds.png',
        'id': '14'
    },
    {
        'image1': '15_default.png',
        'image2': '15_no_mds.png',
        'id': '15'
    },
]


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html', pairs=pairs)


responses = []
response_file = None

try:
    response_file = open('responses.txt', 'a')
except Exception as e:
    print('Error when trying to open file for writing: ', e)

# TODO save response to a file
@app.route('/response', methods=['POST'])
def response():
    winners = {}
    for pair in pairs:
        winner = request.form.get(pair['id'], None)
        if winner:
            winners[pair['id']] = winner
        else:
            raise ValueError('Pair ' + pair['id'] + ' has no winner')

    print(winners)
    responses.append(winners)
    response_file.write(str(winners) + ', ')
    response_file.flush()

    return render_template('submitted.html')

