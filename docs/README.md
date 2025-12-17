to install dependencies, do the following commands : 

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate

pip install -r requirements.txt
```

to launch the application on the cpu, run the following command :

```bash
# if not on the virtual environment, activate it first
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate

python app/main.py
```

to launch the application on the gpu, run the following command :

```bash
# if not on the virtual environment, activate it first
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
python app/main_compute_shader.py
```

