# RecSys

Этот проект содержит несколько
простых реализаций
рекомендательных систем (RS): 
* (1) Content-based Recommender system
* Collaborative Recommender system: 
  * (2) User-Based
  * (3) Item-Based
* (4) Popular Recommender
* (5) Two-Tower Recommender

Dataset: MovieLens 1M Dataset.
### Установка

```basg
git clone https://github.com/WhiteJohny/Pringe.git
cd Pringe
git checkout RecSys
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
### Ознакомление с функционалом RecSys

Файл main.py служит для демонстрации того, 
как работают определенные подходы RecSys. 
Для этого выполните следующее:
```bash
python main.py
```
Будет получен примерно такой результат:

![img.png](media/img.png)
