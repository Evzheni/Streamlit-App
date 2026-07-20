from __future__ import annotations
import pandas as pd
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    model = load(filename)
    y = model.predict(X)
    return y

def load_and_predict_classifier(features: ArrayLike, filename: str) -> tuple[int, ArrayLike]:
    model = load(filename)
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    return prediction, probabilities

def render_regression_section():
    st.header("🪐 Зупинка 1. Планета Лінійної Регресії")
    st.write("Ви прибули на одну з найстаріших планет галактики машинного навчання. Тут усе дуже прямолінійно. Жителі цієї планети люблять знаходити залежності між величинами та передбачати майбутні значення. Вони вірять, що за допомогою правильно підібраної лінії можна описати зв'язок між даними та зробити точний прогноз.")
    st.write("**Лінійна регресія** – одна з найпростіших і водночас найважливіших моделей машинного навчання. Вона використовується для прогнозування числових значень на основі наявних даних. Модель намагається знайти таку пряму (або площину для більшої кількості ознак), яка найкраще описує залежність між вхідними даними та результатом.")
    st.write("Загальна лінійна регресійна модель має вигляд:")
    st.latex(r"y = w_0 + w_1x")
    st.write("""
    де:
    - $x$ – вхідна ознака;
    - $y$ – прогнозоване значення;
    - $w_0$ – вільний член (зміщення);
    - $w_1$ – коефіцієнт, який показує вплив ознаки на результат.
    """)
    st.write("Якщо модель використовує декілька ознак, рівняння набуває вигляду:")
    st.latex(r"y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n")
    st.write("Під час навчання модель підбирає значення коефіцієнтів w так, щоб її прогнози були якомога ближчими до реальних значень. Для цього вона порівнює передбачені значення з фактичними даними та намагається мінімізувати помилку прогнозу.")
    st.write("Для оцінки якості роботи моделі часто використовується середньоквадратична похибка (Mean Squared Error, MSE). Вона показує, наскільки в середньому прогнози моделі відрізняються від реальних значень:")
    st.latex(r"MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2")
    st.write("""
    де:
    - $y_i$ – реальне значення;
    - $\\hat{y}_i$ – прогнозоване значення моделі;
    - $n$ – кількість прикладів у наборі даних.
    """)

    input_feature = st.slider("Вхідна ознака для прогнозу", min_value=-3.0, max_value=3.0, value=0.0, step=0.01)
    
    if st.button("Прогнозувати значення"):
        prediction = load_and_predict([[input_feature]])
        st.write(f"Прогноз: **{prediction[0]:.4f}**")
        visualize_difference(input_feature, prediction[0])

def render_spam_filter_section():
    st.header("🪐 Зупинка 2: Планета Наївного Баєсу")
    st.write("На цій планеті живуть жителі, які чудово вміють працювати з текстом та класифікувати інформацію. **Наївний Баєс** – це алгоритм машинного навчання, який використовується для задач класифікації, тобто визначення, до якої категорії належить певний об'єкт. Незважаючи на свою простоту, ця модель часто застосовується для аналізу текстів, фільтрації спаму, визначення тональності повідомлень та інших задач, де важливо швидко знаходити закономірності у великих наборах даних.")
    st.write("Основна ідея алгоритму базується на теоремі Баєса, яка дозволяє обчислити ймовірність того, що об'єкт належить до певного класу:")
    st.latex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}")
    st.write(r"""
    де:
    - $P(A|B)$ – ймовірність події $A$ за умови, що подія $B$ вже відбулася;
    - $P(B|A)$ – ймовірність спостереження $B$ за умови класу $A$;
    - $P(A)$ – початкова ймовірність класу;
    - $P(B)$ – загальна ймовірність спостереження.
    """)
    st.write("У назві алгоритму є слово «наївний», оскільки модель робить спрощене припущення: вона вважає, що всі ознаки незалежні одна від одної. Наприклад, під час аналізу листа модель оцінює окремі слова та їхній зв'язок із певним класом, не враховуючи складні залежності між ними. Попри це припущення, алгоритм часто показує хороші результати на практичних задачах.")
    st.write("Щоб побачити Наївний Баєс у дії, ми створили власну модель фільтрації спаму. Її завдання – аналізувати текст електронного листа та визначати, чи є він звичайним повідомленням (ham) або небажаним листом (spam).")
    st.write("Спробуйте ввести власний текст листа та перевірте, як модель його класифікує. Вона проаналізує слова у повідомленні та розрахує, до якого класу воно має найбільшу ймовірність належати.")
    email_text = st.text_area("Текст листа:", "WINNER! You have been selected to receive a free gift...")

    if st.button("Перевірити на Спам"):
        prediction, probabilities = load_and_predict_classifier(email_text, "spam_naive_bayes.joblib")
        
        if prediction == 1:
            st.error("Вірогідно, що це СПАМ.")
        else:
            st.success("Вірогідно, що це ЗВИЧАЙНИЙ ЛИСТ.")
            
        render_text_probabilities(probabilities, ['Не спам', 'Спам'])

def render_spam_analysis():
    st.header("Оцінка моделі Наївного Баєсу (спам-фільтру)")
    
    model = load("spam_naive_bayes.joblib")
    X_test = load("X_test_spam.joblib")
    y_test = load("y_test_spam.joblib")
    
    predictions = model.predict(X_test)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, predictions):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, predictions):.3f}")
    col4.metric("F1-score", f"{f1_score(y_test, predictions):.3f}")
    
    st.write("**Матриця плутанини**")
    spacer1, col_cm_plot, spacer2 = st.columns([1, 2, 1])
    with col_cm_plot:
            fig_cm, ax_cm = plt.subplots(figsize=(2, 2))
            cm = confusion_matrix(y_test, predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Не спам', 'Спам'])
            disp.plot(cmap='Greens', ax=ax_cm, colorbar=False)
            ax_cm.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
        
def render_cardio_section():
    st.header("🪐 Зупинка 3. Планета Випадкових лісів")
    st.write("""
       Ви прибули на планету, де рішення приймаються не одним жителем, а цілим лісом жителів. 
    **Random Forest (випадковий ліс)** – це алгоритм машинного навчання, який об'єднує велику кількість моделей дерева рішень. Кожне дерево аналізує дані за власними правилами, а фінальний результат визначається шляхом голосування всіх дерев.

    Такий підхід допомагає зробити модель більш стабільною та зменшити ризик помилок, які можуть виникати в окремих деревах. Random Forest широко використовується для задач класифікації та прогнозування, коли потрібно знаходити складні залежності між багатьма параметрами.

    Основна ідея моделі полягає в тому, що результат формується на основі прогнозів окремих дерев:

    """)

    st.latex(r"\hat{y} = \operatorname{mode}\{T_1(x), T_2(x), ..., T_n(x)\}")

    st.write(r"""
    де:
    - $T_1, T_2, ..., T_n$ – окремі дерева рішень у складі випадкового лісу;
    - $n$ – кількість дерев у моделі;
    - $\hat{y}$ – фінальний прогноз моделі.

    Для задач класифікації кожне дерево робить власний прогноз, а модель обирає клас, який отримав найбільшу кількість голосів.
    """)

    st.write("""
    Ми створили модель на основі Random Forest, яка оцінює ризик серцево-судинних захворювань на основі ваших введених параметрів.

    Введіть свої дані у форму нижче, і модель проаналізує комбінацію характеристик та визначить ймовірний клас ризику.

    Модель враховує взаємозв'язки між різними параметрами та використовує досвід багатьох дерев рішень для формування прогнозу.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_years = st.slider("Вік років", 20, 100, 50)
        gender = st.radio("Стать", options=[1, 2], format_func=lambda x: "Жіноча" if x == 1 else "Чоловіча")
        height = st.slider("Зріст см", 120, 220, 165)
        weight = st.slider("Вага кг", 40.0, 150.0, 70.0, step=0.5)
        
    with col2:
        ap_hi = st.slider("Верхній тиск систолічний", 80, 220, 120)
        ap_lo = st.slider("Нижній тиск діастолічний", 50, 150, 80)
        cholesterol = st.selectbox("Холестерин", options=[1, 2, 3], format_func=lambda x: "Норма" if x == 1 else "Вище норми" if x == 2 else "Значно вище норми")
        gluc = st.selectbox("Глюкоза", options=[1, 2, 3], format_func=lambda x: "Норма" if x == 1 else "Вище норми" if x == 2 else "Значно вище норми")
        
    st.write("Стиль життя:")
    col3, col4, col5 = st.columns(3)
    with col3:
        smoke = 1 if st.checkbox("Куріння") else 0
    with col4:
        alco = 1 if st.checkbox("Вживання алкоголю") else 0
    with col5:
        active = 1 if st.checkbox("Фізична активність", value=True) else 0

    bmi = weight / ((height / 100) ** 2)
    st.info(f"Автоматично розрахований Індекс Маси Тіла ІМТ: **{bmi:.1f}**")

    if st.button("Оцінити ризик"):
        features = [gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years, bmi]
        
        prediction, probabilities = load_and_predict_classifier(features, "rf_cardio_model.joblib")
        
        if prediction == 1:
            st.warning("Модель вказує на наявність ризику серцево-судинних захворювань.")
        else:
            st.success("Ризик серцево-судинних захворювань не виявлено.")
        
        render_text_probabilities(probabilities, ['Відсутній ризик', 'Наявний ризик'])
            

def render_model_analysis():
    st.header("Оцінка моделі Random Forest")

    model = load("rf_cardio_model.joblib")
    X_test = load("X_test_cardio.joblib")
    y_test = load("y_test_cardio.joblib")
    
    predictions = model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)[:, 1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, predictions):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, predictions):.3f}")
    col4.metric("F1-score", f"{f1_score(y_test, predictions):.3f}")
    col5.metric("ROC AUC", f"{roc_auc_score(y_test, predictions_proba):.3f}")
    
    col_plot1, col_plot2, col_plot3 = st.columns(3)
    
    with col_plot1:
        st.write("**Матриця плутанини**")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Немає', 'Є'])
        disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
        plt.tight_layout()
        st.pyplot(fig_cm)
        
    with col_plot2:
        st.write("**Важливість ознак**")
        fig_fi, ax_fi = plt.subplots(figsize=(4, 3))
        importances = model.feature_importances_
        feature_names = X_test.columns
        indices = np.argsort(importances)
        ax_fi.barh(range(len(indices)), importances[indices], color='teal', align='center')
        ax_fi.set_yticks(range(len(indices)))
        ax_fi.set_yticklabels(np.array(feature_names)[indices])
        plt.tight_layout()
        st.pyplot(fig_fi)

    with col_plot3:
        st.write("**ROC Крива**")
        fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
        ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax_roc.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_roc)

    st.write("""
Отримані результати показують, що модель має достатню здатність знаходити закономірності у даних, однак її точність не є максимальною.
Однією з причин є особливості використаного набору даних: він містить викиди та шумові значення, які можуть впливати на якість прогнозування.
Покращення результатів потребувало б додаткового етапу підготовки даних, зокрема Feature Engineering, аналізу та обробки викидів, відбору найбільш важливих ознак і оптимізації параметрів моделі.
Проте основною метою цього завдання було продемонструвати принцип роботи Random Forest та можливості його застосування для задач класифікації, а не повна оптимізація моделі.
""")
            

def create_streamlit_app():
    st.title("Machine Learning Journey 🚀")
    st.write("Ласкаво просимо у подорож галактикою Машинного Навчання!")
    st.write("**Машинне навчання** – це галактика великого всесвіту штучного інтелекту, а саме напрям, що дозволяє комп'ютерам навчатися на основі даних, знаходити закономірності та приймати рішення без явного програмування кожного кроку. Сьогодні ця технологія допомагає розпізнавати зображення, перекладати тексти, рекомендувати фільми, керувати безпілотними автомобілями та вирішувати безліч інших реальних завдань.")
    st.write("Під час цієї невеликої подорожі ми відвідаємо кілька планет галактики машинного навчання, а саме познайомимося з кількома популярними моделями машинного навчання, дізнаємося, як вони працюють, для яких задач використовуються, а також зможемо взаємодіяти з ними та побачити їх у дії.")
    st.write("Тож пристебніть паски, підготуйтеся до старту та рушаймо!")

    st.markdown("---")
    
    render_regression_section()
    st.markdown("---")
    
    render_spam_filter_section()
    st.markdown("---")
    
    render_spam_analysis()
    st.markdown("---")
   
    render_cardio_section()
    st.markdown("---")
    
    render_model_analysis()

    st.markdown("---")
    st.title("Оцінка моделей класифікації")
    st.write("""


Щоб оцінити, наскільки добре працюють моделі машинного навчання, використовуються спеціальні метрики якості. 
Вони дозволяють зрозуміти не лише загальну точність моделі, а й те, які помилки вона робить.

Для моделей **Наївного Баєса** та **Random Forest** ми використовуємо метрики класифікації, оскільки ці алгоритми визначають, до якого класу належить об'єкт (наприклад, спам або не спам, наявність або відсутність ризику захворювання).

До цих метрик належать:
- Accuracy – загальна частка правильних передбачень;
- Precision – точність позитивних прогнозів;
- Recall – здатність знаходити всі позитивні випадки;
- F1-score – баланс між Precision та Recall;
- ROC AUC – здатність моделі правильно розділяти класи.

Для **лінійної регресії** ці метрики не використовуються, оскільки вона вирішує іншу задачу – прогнозування числових значень, а не класифікацію об'єктів.
""")


def render_text_probabilities(probabilities: ArrayLike, labels: list[str]):
    prob_0 = probabilities[0] * 100
    prob_1 = probabilities[1] * 100
    
    st.markdown("**Оцінка моделі:**")
    st.markdown(f"<span style='color:#2e7b32; font-size: 18px;'><b>{labels[0]}:</b> {prob_0:.1f}%</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#d32f2f; font-size: 18px;'><b>{labels[1]}:</b> {prob_1:.1f}%</span>", unsafe_allow_html=True)

def visualize_difference(input_feature: float, prediction: ArrayLike):
    X_filename = "X.joblib"
    y_filename = "y.joblib"
    X = load(X_filename)
    y = load(y_filename)

    actual_target = y[_index_of_closest(X, input_feature)]
    difference = actual_target - prediction

    fig = plt.figure(figsize=(8, 5))
    
    plt.scatter(X, y, color='grey', label='Dataset', alpha=0.6)
    plt.scatter(input_feature, actual_target, color='blue', label='Actual Target', zorder=5, s=80)
    plt.scatter(input_feature, prediction, color='red', label='Predicted Target', zorder=5, s=80)
    
    plt.plot([input_feature, input_feature], [actual_target, prediction], 'k--', label='Difference')
    
    text_y_pos = (actual_target + prediction) / 2
    plt.annotate(f'Difference = {difference:.2f}', (input_feature + 0.1, text_y_pos), fontsize=10)

    plt.title("Prediction vs Actual Target")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    st.pyplot(fig)

def _index_of_closest(X: ArrayLike, k: float) -> int:
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx

if __name__ == '__main__':
    create_streamlit_app()