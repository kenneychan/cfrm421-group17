# Ensemble ML -- Setup Guide

## Prerequisites
Make sure Python 3.10+ is installed.
```bash
python --version
```

---

## Step 1 -- Clone the repository
```bash
git clone <repository-url>
cd ensemble_ml
```

---

## Step 2 -- Create the virtual environment
```bash
python -m venv venv
```

---

## Step 3 -- Activate the virtual environment

Mac/Linux:
```bash
source venv/bin/activate
```
Windows:
```bash
venv\Scripts\activate
```
You'll know it's active when you see `(venv)` at the start of your terminal prompt.

---

## Step 4 -- Upgrade pip and install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 5 -- Run the project
```bash
python main.py
```

---

## Step 6 -- Deactivate when done
```bash
deactivate
```

---

## Project structure
```
ensemble_ml/
├── model.py          # BaseModel -- data fetching + 80/10/10 split (do not edit)
├── student_models.py # RandomModel control baseline          (do not edit)
├── controller.py     # run_ensemble() -- majority vote       (do not edit)
├── main.py           # Entry point                           (do not edit)
├── student1.py       # Student 1 -- edit this file only
├── student2.py       # Student 2 -- edit this file only
├── student3.py       # Student 3 -- edit this file only
├── student4.py       # Student 4 -- edit this file only
├── student5.py       # Student 5 -- edit this file only
├── requirements.txt
├── .gitignore
└── setup.md
```

---

## Your only job
Open **your** `studentN.py` file and fill in `predict()`:

```python
class Student1Model(BaseModel):
    name = "My Model Name"

    def predict(self, data: pd.DataFrame) -> int:
        # use self.train    -> fit your model
        # use self.refine   -> tune hyperparameters
        # use self.evaluate -> final holdout (touch last!)
        return 1   # or -1
```

---

## Tips
- Only edit your own `studentN.py` -- do not touch anyone else's file
- Never commit the `venv/` folder to Git -- it's already in `.gitignore`
- Each team member runs these steps once on their own machine
- Next time you return to the project, just re-activate (Step 3)
