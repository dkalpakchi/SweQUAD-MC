# SweQUAD-MC
The official repository for SweQUAD-MC dataset - the first Swedish dataset of multiple-choice questions (MCQs) for reading comprehension. The repository also contains the original implementation of our distractor generation models, described in the paper "BERT-based distractor generation for Swedish reading comprehension questions using a small-scale dataset".

## Information about SweQUAD-MC
The original split into training, development and test sets is available in the `data` folder. Descriptive statistics of all splits can be found in the original paper. Each set is a JSON file with each MCQ having the following format (note that the ``context`` is truncated only for presentation purposes):
```json
{
    "choices": [
        {
            "end": 1296,
            "extra": null,
            "start": 1290,
            "text": "100 kr",
            "type": "Correct answer"
        },
        {
            "end": 1184,
            "extra": {
                "comment": "3 kr"
            },
            "start": 1183,
            "text": "3",
            "type": "Distractor"
        },
        {
            "end": 78,
            "extra": {
                "comment": "2017 kr"
            },
            "start": 74,
            "text": "2017",
            "type": "Distractor"
        }
    ],
    "context": "Företagets ansvar\nDen nya elsäkerhetslagen, som trädde i kraft den 1 juli 2017, innebär att elinstallationsföretagen har fått ett helt nytt ansvar.\nFöretagen omfattas av följande regler:\nAlla företag ska känna till vilka regler som gäller och se till att företaget följer dem. [...] Mer information i handboken\nMer information om företagaransvaret finns i avsnitt 3 i Elsäkerhetsverket handbok om elinstallationsarbete enligt elsäkerhetslagen. Boken går att beställa för 100 kr/st i vår publikationsshop. Du kan även ladda hem den som pdf utan kostnad, se länk längre ner på sidan.",
    "question": "Hur mycket kostar Elsäkerhetsverkets handbok?"
}
```
``start`` and ``end`` fields are the character positions in the ``context``, where each alternative can be found, i.e., ``context[start:end]``. Note that it was required that both the correct answer and all distractors exist in the text, however, if slight reformulations are needed (e.g., changing of grammatical form), there is the field ``extra``. If ``extra`` is ``null``, then no reformulation was deemed necessary, otherwise, the reformulation is given in the ``comment`` field.

**IMPORTANT**
If you find any typos, mistakes or other inconsistencies, please contact the first author of the paper at ``dmytroka@kth.se``.

## How to use models
Example of using the generator:
```
python3 generator.py -f checkpoint-3500 -dg dg_args.bin -o res_dev_nlo_min_first.txt -d ../../../data/dev.json -nlo -ls min_first
```