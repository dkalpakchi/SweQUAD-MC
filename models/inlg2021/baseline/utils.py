import json

def get_text(c, field):
    return c["extra"][field].strip() if c["extra"] else c["text"].strip()

def get_correct_answer(choices):
    return list(filter(lambda x: x['type'] == 'Correct answer', choices))[0]

def get_distractors(choices):
    return list(filter(lambda x: x['type'] == 'Distractor', choices))

def read_dataset_from_files(fnames, lang='en'):
    answer_transformations = {}
    total, commented = 0, 0
    for fn in fnames:
        d = json.load(open(fn))
        assert 'data' in d, "Not compatible with Textinator format"

        for dp in d['data']:
            context, question, choices = dp['context'], dp['question'], dp['choices']
            ca_choice = get_correct_answer(choices)
            ca = get_text(ca_choice, 'comment')
            dis = [get_text(d, 'comment') for d in get_distractors(dp['choices'])]
            answer = {
                'text': ca,
                'context': ca_choice['context']
            }
            yield question, answer, context, dis