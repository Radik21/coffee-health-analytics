import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(
    page_title="Coffee Health Analytics Pro",
    page_icon="☕",
    layout="wide"
)


# Вспомогательные функции
def get_metric_label(metric):
    labels = {
        'Sleep_Hours': 'Продолжительность сна (часов)',
        'BMI': 'Индекс массы тела',
        'Heart_Rate': 'Сердечный ритм (уд/мин)',
        'Physical_Activity_Hours': 'Физическая активность (часов/неделю)',
        'Coffee_Intake': 'Потребление кофе (чашек/день)',
        'Age': 'Возраст (лет)',
        'Caffeine_mg': 'Кофеин (мг/день)',
        'Stress_Level': 'Уровень стресса'
    }
    return labels.get(metric, metric)


def categorize_health_risk(bmi):
    if bmi < 18.5:
        return 'Недостаточный вес'
    elif 18.5 <= bmi < 25:
        return 'Нормальный вес'
    elif 25 <= bmi < 30:
        return 'Избыточный вес'
    else:
        return 'Ожирение'


def categorize_sleep_quality(hours):
    if hours >= 7:
        return 'Отличный'
    elif 6 <= hours < 7:
        return 'Хороший'
    elif 5 <= hours < 6:
        return 'Удовлетворительный'
    else:
        return 'Недостаточный'


def preprocess_data(df):
    """Предобработка данных для ML"""
    df_processed = df.copy()

    # Кодируем категориальные переменные
    categorical_columns = ['Gender', 'Stress_Level', 'Sleep_Quality', 'Country']

    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))

    return df_processed


# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv('coffee_health_10000.csv')


df = load_data()

# Предобработка данных
df_processed = preprocess_data(df)

if 'Sleep Quality' in df.columns:
    sleep_quality_map = {'Good': 2, 'Fair': 1, 'Poor': 0}
    df['Sleep_Quality_Score'] = df['Sleep Quality'].map(sleep_quality_map)
    df_processed['Sleep_Quality_Score'] = df_processed['Sleep Quality'].map(sleep_quality_map)

# Sidebar с расширенными настройками
st.sidebar.header("⚙️ Параметры анализа")

selected_country = st.sidebar.selectbox("Выберите страну", ['Все'] + list(df['Country'].unique()))
min_age, max_age = st.sidebar.slider("Возрастной диапазон",
                                     int(df['Age'].min()),
                                     int(df['Age'].max()),
                                     (25, 60))
coffee_range = st.sidebar.slider("Диапазон потребления кофе (чашек/день)",
                                 float(df['Coffee_Intake'].min()),
                                 float(df['Coffee_Intake'].max()),
                                 (1.0, 5.0))

# Дополнительные фильтры
st.sidebar.subheader("Дополнительные фильтры")
selected_gender = st.sidebar.multiselect("Пол", options=['Male', 'Female'], default=['Male', 'Female'])
stress_levels = st.sidebar.multiselect("Уровень стресса",
                                       options=['Low', 'Medium', 'High'],
                                       default=['Low', 'Medium', 'High'])

# Фильтрация данных
filtered_df = df.copy()
filtered_df_processed = df_processed.copy()

if selected_country != 'Все':
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    filtered_df_processed = filtered_df_processed[filtered_df_processed['Country'] == selected_country]

if selected_gender:
    filtered_df = filtered_df[filtered_df['Gender'].isin(selected_gender)]
    filtered_df_processed = filtered_df_processed[filtered_df_processed['Gender'].isin(selected_gender)]

if stress_levels:
    filtered_df = filtered_df[filtered_df['Stress_Level'].isin(stress_levels)]
    filtered_df_processed = filtered_df_processed[filtered_df_processed['Stress_Level'].isin(stress_levels)]

filtered_df = filtered_df[
    (filtered_df['Age'] >= min_age) &
    (filtered_df['Age'] <= max_age) &
    (filtered_df['Coffee_Intake'] >= coffee_range[0]) &
    (filtered_df['Coffee_Intake'] <= coffee_range[1])
    ]

filtered_df_processed = filtered_df_processed[
    (filtered_df_processed['Age'] >= min_age) &
    (filtered_df_processed['Age'] <= max_age) &
    (filtered_df_processed['Coffee_Intake'] >= coffee_range[0]) &
    (filtered_df_processed['Coffee_Intake'] <= coffee_range[1])
    ]

# Основной интерфейс
st.title("☕ Продвинутый анализ взаимосвязи потребления кофе и показателей здоровья")
st.markdown("---")

# Расширенные ключевые метрики
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Объем выборки", f"{len(filtered_df):,}",
              delta=f"{len(filtered_df) - 1000}" if len(filtered_df) != 1000 else None)
with col2:
    avg_coffee = filtered_df['Coffee_Intake'].mean()
    st.metric("Среднее потребление кофе", f"{avg_coffee:.1f} чашек/день")
with col3:
    avg_sleep = filtered_df['Sleep_Hours'].mean()
    st.metric("Средняя продолжительность сна", f"{avg_sleep:.1f} часов")
with col4:
    coffee_sleep_corr = filtered_df['Coffee_Intake'].corr(filtered_df['Sleep_Hours'])
    st.metric("Корреляция кофе-сон", f"{coffee_sleep_corr:.3f}")

# Второй ряд метрик
col1, col2, col3, col4 = st.columns(4)
with col1:
    health_risk_count = len(filtered_df[filtered_df['BMI'] >= 25])
    st.metric("Риск здоровья (ИМТ ≥ 25)", f"{health_risk_count:,}")
with col2:
    poor_sleep_count = len(filtered_df[filtered_df['Sleep_Hours'] < 6])
    st.metric("Нарушения сна (<6 часов)", f"{poor_sleep_count:,}")
with col3:
    high_coffee_count = len(filtered_df[filtered_df['Coffee_Intake'] > 4])
    st.metric("Высокое потребление (>4 чашек)", f"{high_coffee_count:,}")
with col4:
    avg_heart_rate = filtered_df['Heart_Rate'].mean()
    st.metric("Средний пульс", f"{avg_heart_rate:.1f} уд/мин")

# Визуализации
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Основные зависимости", "📊 Распределения", "🎯 ML-анализ", "👥 Кластеризация", "💡 Рекомендации"])

with tab1:
    st.subheader("Многомерный анализ зависимостей")

    # Интерактивный 3D scatter plot
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_3d = px.scatter_3d(filtered_df,
                               x='Coffee_Intake',
                               y='Sleep_Hours',
                               z='Heart_Rate',
                               color='Age',
                               size='BMI',
                               hover_data=['Stress_Level', 'Physical_Activity_Hours'],
                               title='3D анализ: Кофе vs Сон vs Пульс',
                               labels={
                                   'Coffee_Intake': 'Потребление кофе',
                                   'Sleep_Hours': 'Продолжительность сна',
                                   'Heart_Rate': 'Сердечный ритм'
                               })
        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        # Матрица корреляций
        numeric_cols = ['Coffee_Intake', 'Sleep_Hours', 'BMI', 'Heart_Rate',
                        'Physical_Activity_Hours', 'Age', 'Caffeine_mg']
        corr_matrix = filtered_df[numeric_cols].corr()

        fig_corr = px.imshow(corr_matrix,
                             title='Матрица корреляций',
                             color_continuous_scale='RdBu_r',
                             aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Дополнительные визуализации
    col1, col2 = st.columns(2)

    with col1:
        # Взаимодействие кофе и физической активности
        fig = px.scatter(filtered_df,
                         x='Coffee_Intake',
                         y='Physical_Activity_Hours',
                         color='Sleep_Hours',
                         size='Age',
                         trendline='lowess',
                         title='Кофе vs Физическая активность vs Сон',
                         labels={
                             'Coffee_Intake': 'Потребление кофе (чашек/день)',
                             'Physical_Activity_Hours': 'Физ. активность (часов/неделю)',
                             'Sleep_Hours': 'Продолжительность сна'
                         })
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Влияние кофе на сон в разных возрастных группах
        filtered_df['Age_Group'] = pd.cut(filtered_df['Age'],
                                          bins=[18, 30, 45, 60, 100],
                                          labels=['18-30', '31-45', '46-60', '60+'])

        fig = px.box(filtered_df,
                     x='Age_Group',
                     y='Sleep_Hours',
                     color='Age_Group',
                     title='Влияние возраста на взаимосвязь кофе-сон')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Расширенный статистический анализ")

    col1, col2 = st.columns(2)

    with col1:
        selected_metric = st.selectbox("Выберите показатель для анализа",
                                       ['Sleep_Hours', 'BMI', 'Heart_Rate',
                                        'Physical_Activity_Hours', 'Coffee_Intake', 'Age'])

        # Улучшенная гистограмма с KDE
        fig = px.histogram(filtered_df, x=selected_metric,
                           nbins=30,
                           marginal="box",
                           title=f'Распределение {get_metric_label(selected_metric)}',
                           opacity=0.7)

        # Добавляем статистические аннотации
        mean_val = filtered_df[selected_metric].mean()
        median_val = filtered_df[selected_metric].median()

        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Среднее: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                      annotation_text=f"Медиана: {median_val:.2f}")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Сравнение по полу
        if 'Gender' in filtered_df.columns:
            fig = px.violin(filtered_df,
                            x='Gender',
                            y='Sleep_Hours',
                            color='Gender',
                            box=True,
                            title='Распределение продолжительности сна по полу')
            st.plotly_chart(fig, use_container_width=True)

    # Анализ по странам
    st.subheader("Сравнительный анализ по странам")

    country_stats = filtered_df.groupby('Country').agg({
        'Coffee_Intake': 'mean',
        'Sleep_Hours': 'mean',
        'BMI': 'mean',
        'Heart_Rate': 'mean'
    }).round(2)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(country_stats.style.background_gradient(cmap='Blues'),
                     use_container_width=True)

    with col2:
        fig = px.bar(country_stats,
                     y=country_stats.index,
                     x='Coffee_Intake',
                     title='Среднее потребление кофе по странам',
                     orientation='h')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Продвинутое прогнозное моделирование")

    # Множественное моделирование с предобработанными данными
    models_config = {
        'Продолжительность сна': ('Sleep_Hours', ['Age', 'Coffee_Intake', 'BMI',
                                                  'Physical_Activity_Hours', 'Stress_Level_encoded']),
        'Сердечный ритм': ('Heart_Rate', ['Age', 'Coffee_Intake', 'BMI',
                                          'Physical_Activity_Hours', 'Sleep_Hours', 'Stress_Level_encoded']),
        'ИМТ': ('BMI', ['Age', 'Coffee_Intake', 'Physical_Activity_Hours',
                        'Sleep_Hours', 'Heart_Rate', 'Stress_Level_encoded'])
    }

    selected_model = st.selectbox("Выберите целевую переменную", list(models_config.keys()))

    target, features = models_config[selected_model]

    # Используем предобработанные данные
    available_features = [f for f in features if f in filtered_df_processed.columns]

    if len(available_features) > 0 and target in filtered_df_processed.columns:
        X = filtered_df_processed[available_features]
        y = filtered_df_processed[target]

        # Проверяем, что нет пропущенных значений
        X = X.dropna()
        y = y.loc[X.index]

        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            try:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Расширенные метрики
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MSE", f"{mse:.3f}")
                with col2:
                    st.metric("MAE", f"{mae:.3f}")
                with col3:
                    st.metric("R²", f"{r2:.3f}")
                with col4:
                    rmse = np.sqrt(mse)
                    st.metric("RMSE", f"{rmse:.3f}")

                # Визуализация предсказаний vs реальных значений
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                         name='Предсказания', marker=dict(color='blue')))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                         y=[y_test.min(), y_test.max()],
                                         mode='lines',
                                         name='Идеальная линия',
                                         line=dict(color='red', dash='dash')))
                fig.update_layout(title='Предсказания vs Реальные значения',
                                  xaxis_title='Реальные значения',
                                  yaxis_title='Предсказания')
                st.plotly_chart(fig, use_container_width=True)

                # Важность признаков
                importance_df = pd.DataFrame({
                    'feature': available_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)

                fig = px.bar(importance_df, x='importance', y='feature',
                             title='Важность признаков в модели',
                             labels={'importance': 'Важность признака', 'feature': 'Признак'})
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Ошибка при построении модели: {str(e)}")
                st.info("Попробуйте изменить параметры фильтрации для увеличения размера выборки")

        else:
            st.warning("Недостаточно данных для построения модели. Увеличьте диапазон фильтров.")
    else:
        st.error("Отсутствуют необходимые признаки для построения модели")

with tab4:
    st.subheader("Кластерный анализ пользователей")

    # Используем только числовые колонки для кластеризации
    cluster_features = ['Coffee_Intake', 'Sleep_Hours', 'BMI', 'Heart_Rate',
                        'Physical_Activity_Hours', 'Age']
    cluster_df = filtered_df[cluster_features].dropna()

    if len(cluster_df) > 10:
        # Масштабирование данных
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)

        # Определение оптимального числа кластеров
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)

        # Визуализация метода локтя
        fig_elbow = px.line(x=range(1, 11), y=wcss,
                            title='Метод локтя для определения числа кластеров',
                            labels={'x': 'Число кластеров', 'y': 'WCSS'})
        st.plotly_chart(fig_elbow, use_container_width=True)

        # Кластеризация
        optimal_clusters = st.slider("Выберите число кластеров", 2, 6, 4)
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        cluster_df['Cluster'] = clusters

        col1, col2 = st.columns(2)

        with col1:
            # Визуализация кластеров
            fig = px.scatter(cluster_df,
                             x='Coffee_Intake',
                             y='Sleep_Hours',
                             color='Cluster',
                             size='BMI',
                             hover_data=['Age', 'Heart_Rate'],
                             title='Кластерный анализ: Кофе vs Сон',
                             labels={
                                 'Coffee_Intake': 'Потребление кофе',
                                 'Sleep_Hours': 'Продолжительность сна'
                             })
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Характеристики кластеров
            cluster_stats = cluster_df.groupby('Cluster').mean()
            st.dataframe(cluster_stats.style.background_gradient(cmap='YlOrBr'),
                         use_container_width=True)

        # Интерпретация кластеров
        st.subheader("Интерпретация кластеров")

        # Автоматическая интерпретация на основе характеристик
        for cluster_num in range(optimal_clusters):
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster_num]
            avg_coffee = cluster_data['Coffee_Intake'].mean()
            avg_sleep = cluster_data['Sleep_Hours'].mean()
            avg_bmi = cluster_data['BMI'].mean()

            if avg_coffee < 2 and avg_sleep >= 7:
                description = "Низкое потребление кофе, отличный сон - Здоровый профиль"
            elif avg_coffee > 4 and avg_sleep < 6:
                description = "Высокое потребление кофе, плохой сон - Группа риска"
            elif avg_bmi > 28:
                description = "Высокий ИМТ - Требуется консультация специалиста"
            else:
                description = "Сбалансированный профиль - Умеренные показатели"

            st.write(f"**Кластер {cluster_num}**: {description}")
            st.write(f"   - Среднее потребление кофе: {avg_coffee:.1f} чашек/день")
            st.write(f"   - Средняя продолжительность сна: {avg_sleep:.1f} часов")
            st.write(f"   - Средний ИМТ: {avg_bmi:.1f}")
            st.write("---")

    else:
        st.warning("Недостаточно данных для кластерного анализа")

with tab5:
    st.subheader("Персонализированные рекомендации и экономический анализ")

    # Персонализированные рекомендации
    st.subheader("🏥 Персонализированные рекомендации")

    # Анализ рисков в текущей выборке
    filtered_df['Health_Risk'] = filtered_df['BMI'].apply(categorize_health_risk)
    filtered_df['Sleep_Quality_Cat'] = filtered_df['Sleep_Hours'].apply(categorize_sleep_quality)

    col1, col2 = st.columns(2)

    with col1:
        risk_distribution = filtered_df['Health_Risk'].value_counts()
        fig = px.pie(risk_distribution,
                     values=risk_distribution.values,
                     names=risk_distribution.index,
                     title='Распределение рисков здоровья по ИМТ')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sleep_distribution = filtered_df['Sleep_Quality_Cat'].value_counts()
        fig = px.bar(sleep_distribution,
                     x=sleep_distribution.values,
                     y=sleep_distribution.index,
                     orientation='h',
                     title='Качество сна в выборке')
        st.plotly_chart(fig, use_container_width=True)

    # Экономический анализ
    st.subheader("📊 Расширенный экономический анализ")

    # Более сложные расчеты
    base_population = 1000000  # Условная популяция
    sample_ratio = len(filtered_df) / len(df) if len(df) > 0 else 1
    target_population = int(base_population * sample_ratio)

    # Расчеты экономии
    development_cost = 250000
    maintenance_cost = 50000

    # Предполагаемая экономия на человека
    savings_per_person = {
        'Улучшение сна': 1500,
        'Снижение рисков здоровья': 3000,
        'Повышение продуктивности': 2000
    }

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_savings = sum(savings_per_person.values()) * target_population * 0.1  # 10% эффективность
        st.metric("Годовая экономия", f"{total_savings:,.0f} ₽")

    with col2:
        st.metric("Затраты на разработку", f"{development_cost:,} ₽")

    with col3:
        roi_1_year = ((total_savings - development_cost - maintenance_cost) /
                      (development_cost + maintenance_cost)) * 100
        st.metric("ROI за 1 год", f"{roi_1_year:.0f}%")

    with col4:
        payback_period = (development_cost + maintenance_cost) / total_savings if total_savings > 0 else float('inf')
        st.metric("Окупаемость", f"{payback_period:.1f} лет" if payback_period != float('inf') else "∞")

    # Детализированная разбивка экономии
    st.subheader("Детализация экономического эффекта")

    savings_breakdown = pd.DataFrame({
        'Категория': list(savings_per_person.keys()),
        'Экономия на человека': list(savings_per_person.values()),
        'Общая экономия': [savings_per_person[k] * target_population * 0.1 for k in savings_per_person.keys()]
    })

    fig = px.bar(savings_breakdown,
                 x='Категория',
                 y='Общая экономия',
                 title='Детализация годовой экономии по категориям',
                 color='Категория')
    st.plotly_chart(fig, use_container_width=True)

    # Рекомендации для разных групп
    st.subheader("🎯 Целевые рекомендации для разных групп")

    recommendations_data = {
        'Группа': ['Низкое потребление (<2 чашек)', 'Умеренное (2-4 чашки)', 'Высокое (>4 чашек)'],
        'Рекомендации': [
            'Можно увеличить потребление до 2-3 чашек для улучшения когнитивных функций',
            'Оптимальный уровень, поддерживать текущее потребление',
            'Рекомендуется снизить потребление, особенно во второй половине дня'
        ],
        'Приоритет': ['Низкий', 'Средний', 'Высокий']
    }

    rec_df = pd.DataFrame(recommendations_data)
    st.dataframe(rec_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Методология**: Анализ основан на данных о 10,000 респондентах. 
Все расчеты носят демонстрационный характер.
""")
