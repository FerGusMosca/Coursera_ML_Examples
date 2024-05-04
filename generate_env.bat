echo install virtualenv
call pip install virtualenv

echo Create new virtual environment
call virtualenv venv

echo activate current environment
call .\venv\Scripts\activate

echo Install all modules from requirements file
call pip install -r requirements.txt
