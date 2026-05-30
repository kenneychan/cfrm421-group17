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

## Step 5 -- Copy the notebook for your own use
Each student must make their own personal copy of the notebook.
Do NOT edit `student_dev.ipynb` directly -- it is shared by the whole team.

```bash
cp student_dev.ipynb student1_dev.ipynb   # replace 1 with your student number
```

Then open your personal copy:
```bash
jupyter notebook student1_dev.ipynb
```

Set `STUDENT_NUMBER` at the top of the notebook to your student number and you are ready to go.

> **Important:** Always use *Kernel → Restart Kernel and Run All Cells* after
> making changes to your `studentN.py` file.

---

## Step 6 -- Add your notebook to .gitignore
Your personal notebook is scratch work and should not be committed to the repo.
Add it to `.gitignore`:
```
studentN_dev.ipynb     # add your own copy here, e.g. student1_dev.ipynb
.ipynb_checkpoints/
```

---

## Step 7 -- Run the full ensemble
Once your model is implemented and tested in your notebook, run the full team ensemble:
```bash
python main.py
```

---

## Step 8 -- Deactivate when done
```bash
deactivate
```

---

## Project structure
```
ensemble_ml/
├── model.py              # BaseModel -- data fetching + 80/10/10 split (do not edit)
├── control.py            # RandomModel control baseline                 (do not edit)
├── controller.py         # run_ensemble() -- majority vote              (do not edit)
├── main.py               # Entry point                                  (do not edit)
├── student_dev.ipynb     # Shared notebook template                     (do not edit)
├── student1.py           # Student 1 -- edit this file only
├── student2.py           # Student 2 -- edit this file only
├── student3.py           # Student 3 -- edit this file only
├── student4.py           # Student 4 -- edit this file only
├── student5.py           # Student 5 -- edit this file only
├── requirements.txt
├── .gitignore
└── setup.md
```

---

## Your only job
1. Copy `student_dev.ipynb` to `studentN_dev.ipynb` (your personal scratch notebook)
2. Open your personal notebook and set `STUDENT_NUMBER` to your number
3. Use the scratch cells to prototype your algorithm
4. Copy finished logic into your `studentN.py` file
5. Restart the kernel and run all cells to verify

```python
class Student1Model(BaseModel):
    name = "My Model Name"

    def train(self) -> None:
        # fit your model using self.train_set (80%)

    def refine_model(self) -> None:
        # tune using self.refine_set (10%)

    def predict(self) -> int:
        # return 1 (UP) or -1 (DOWN) using self.evaluate_set (10%)
```

---

## Tips
- Only edit your own `studentN.py` -- do not touch anyone else's file
- Never commit `venv/` or your personal notebook -- both are in `.gitignore`
- Each team member runs the setup steps once on their own machine
- Next time you return to the project, just re-activate (Step 3)
