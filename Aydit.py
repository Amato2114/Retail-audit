import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import io
import logging

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)  # Подавить логи Prophet

# Настройка страницы
st.set_page_config(page_title="RetailLoss Sentinel v8", layout="wide", page_icon="🛡️")

# Заголовок
st.title("🛡️ RetailLoss Sentinel v8")
st.markdown("**Инновационный AI-анализатор потерь для гипермаркетов**")
st.markdown("What-if • Pareto • ABC • Прогноз по категориям • Улучшенный ML")

# Сайдбар
with st.sidebar:
    st.header("📁 Загрузка данных")
    uploaded_file = st.file_uploader("Выберите файл Excel", type=["xlsx"])
    
    if st.button("🔄 Тестовые данные (300 строк)"):
        st.session_state.use_test = True
        st.rerun()
    
    st.markdown("---")
    st.success("✅ Все библиотеки готовы!")

# Кэшированные тестовые данные
@st.cache_data
def сгенерировать_тестовые_данные():
    np.random.seed(42)
    сегодня = datetime.now()
    даты = pd.date_range(end=сегодня, periods=300, freq='D')
    категории = np.random.choice(['Молочка', 'Мясо', 'Овощи', 'Алкоголь', 'Хлеб', 'Бакалея', 'Заморозка'], size=300)
    суммы_потерь = np.random.uniform(300, 7000, size=300).round(2)
    магазины = np.random.choice(['Магазин1', 'Магазин2', 'Магазин3', 'Магазин4', 'Магазин5'], size=300)
    df = pd.DataFrame({
        'Дата': даты,
        'Категория': категории,
        'СуммаПотерь': суммы_потерь,
        'Магазин': магазины
    })
    df['Дата'] = df['Дата'].dt.strftime('%d.%m.%Y')
    return df

# Авто-распознавание колонок
def detect_columns(df):
    df = df.copy()
    lower_columns = {col.lower(): col for col in df.columns}
    
    date_candidates = [col for col in lower_columns if 'дат' in col or 'date' in col]
    category_candidates = [col for col in lower_columns if 'кат' in col or 'cat' in col or 'товар' in col or 'продукт' in col]
    loss_candidates = [col for col in lower_columns if 'пот' in col or 'loss' in col or 'сум' in col or 'убыт' in col or 'спис' in col]
    store_candidates = [col for col in lower_columns if 'маг' in col or 'store' in col or 'филиал' in col]
    
    if not date_candidates:
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                if parsed.notna().mean() > 0.7:
                    date_candidates = [col.lower()]
                    break
            except:
                pass
    if not category_candidates:
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.1 and df[col].str.len().mean() > 3:
                category_candidates = [col.lower()]
                break
    if not loss_candidates:
        for col in df.columns:
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.notna().mean() > 0.9 and numeric.mean() > 100:
                loss_candidates = [col.lower()]
                break
    if not store_candidates:
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.05:
                store_candidates = [col.lower()]
                break
    
    if not all([date_candidates, category_candidates, loss_candidates, store_candidates]):
        st.error("❌ Не удалось автоматически распознать все обязательные колонки.")
        st.stop()
    
    date_col = lower_columns[date_candidates[0]]
    category_col = lower_columns[category_candidates[0]]
    loss_col = lower_columns[loss_candidates[0]]
    store_col = lower_columns[store_candidates[0]]
    
    st.success(f"✅ Распознаны колонки:\n- Дата → **{date_col}**\n- Категория → **{category_col}**\n- СуммаПотерь → **{loss_col}**\n- Магазин → **{store_col}**")
    
    df = df.rename(columns={date_col: 'Дата', category_col: 'Категория', loss_col: 'СуммаПотерь', store_col: 'Магазин'})
    return df

# Загрузка данных
if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = xls.sheet_names[0]
        df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
        df_raw = detect_columns(df_raw)
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {str(e)}")
        st.stop()
elif st.session_state.get('use_test', False):
    df_raw = сгенерировать_тестовые_данные()
    df_raw['Дата'] = pd.to_datetime(df_raw['Дата'], format='%d.%m.%Y')
    st.session_state.use_test = False
else:
    st.info("👆 Загрузите файл Excel или нажмите «Тестовые данные» для демо.")
    preview_df = сгенерировать_тестовые_данные()
    st.dataframe(preview_df.head(20), width='stretch')
    st.stop()

# Кнопки сброса и оптимального
with st.sidebar:
    st.markdown("---")
    col_reset, col_optimal = st.columns(2)
    with col_reset:
        if st.button("🔄 Сброс"):
            for key in ['reduce_a', 'reduce_peak', 'reduce_top_store']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col_optimal:
        if st.button("🎯 Оптимальный"):
            st.session_state.reduce_a = 25
            st.session_state.reduce_peak = 20
            st.session_state.reduce_top_store = 30
            st.rerun()

# Фильтры
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 Фильтры")
    магазины_список = ['Все'] + sorted(df_raw['Магазин'].unique().tolist())
    выбранные_магазины = st.multiselect("Магазины", магазины_список, default='Все')
    категории_список = ['Все'] + sorted(df_raw['Категория'].unique().tolist())
    выбранные_категории = st.multiselect("Категории", категории_список, default='Все')
    min_date = df_raw['Дата'].min().date()
    max_date = df_raw['Дата'].max().date()
    выбранный_период = st.date_input("Период дат", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    st.markdown("---")
    st.subheader("🧮 What-if сценарии")
    reduce_a = st.slider("Снижение в A-классе, %", 0, 50, value=st.session_state.get('reduce_a', 10))
    reduce_peak = st.slider("Снижение в пиковые дни, %", 0, 50, value=st.session_state.get('reduce_peak', 15))
    reduce_top_store = st.slider("Снижение в топ-магазине, %", 0, 50, value=st.session_state.get('reduce_top_store', 20))

# Применение фильтров
df = df_raw.copy()
if 'Все' not in выбранные_магазины:
    df = df[df['Магазин'].isin(выбранные_магазины)]
if 'Все' not in выбранные_категории:
    df = df[df['Категория'].isin(выбранные_категории)]
df = df[(df['Дата'].dt.date >= выбранный_период[0]) & (df['Дата'].dt.date <= выбранный_период[1])]

if df.empty:
    st.warning("⚠️ Нет данных по выбранным фильтрам.")
    st.stop()

# Основные расчёты
текущие_потери = df['СуммаПотерь'].sum()

суммарные_потери = df.groupby('Категория')['СуммаПотерь'].sum().reset_index().sort_values('СуммаПотерь', ascending=False)
потери_по_магазинам = df.groupby('Магазин')['СуммаПотерь'].sum().reset_index().sort_values('СуммаПотерь', ascending=False)

abc = суммарные_потери.copy()
abc['Доля_%'] = (abc['СуммаПотерь'] / текущие_потери * 100).round(2)
abc['Накопительная_доля'] = abc['Доля_%'].cumsum()
abc['ABC'] = abc['Накопительная_доля'].apply(lambda x: 'A' if x <= 80 else 'B' if x <= 95 else 'C')

pareto_store = потери_по_магазинам.copy()
pareto_store['Доля_%'] = (pareto_store['СуммаПотерь'] / текущие_потери * 100).round(2)
pareto_store['Накопительная_доля'] = pareto_store['Доля_%'].cumsum()
pareto_store['Pareto'] = pareto_store['Накопительная_доля'].apply(lambda x: '80%' if x <= 80 else '95%' if x <= 95 else '100%')

a_class_loss = abc[abc['ABC'] == 'A']['СуммаПотерь'].sum()
экономия_a = round(a_class_loss * reduce_a / 100)

df_day = df.copy()
day_map = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}
df_day['День'] = df_day['Дата'].dt.weekday.map(day_map)
peak_days_loss = df_day.groupby('День')['СуммаПотерь'].sum().nlargest(2).sum()
экономия_peak = round(peak_days_loss * reduce_peak / 100)

top_store_loss = pareto_store[pareto_store['Pareto'] == '80%']['СуммаПотерь'].sum()
экономия_store = round(top_store_loss * reduce_top_store / 100)

общая_экономия = экономия_a + экономия_peak + экономия_store

# Большая метрика
st.markdown(f"""
    <div style='text-align: center; margin: 30px 0;'>
        <h1 style='color: #ef4444; font-size: 48px; margin: 0;'>
            {текущие_потери:,.0f} ₽
        </h1>
        <p style='font-size: 20px; color: gray; margin: 5px 0;'>
            Общие потери за выбранный период
        </p>
    </div>
""", unsafe_allow_html=True)

# What-if результаты
st.markdown("### 💰 Потенциальная экономия")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("A-класс", f"{экономия_a:,.0f} ₽", delta=f"-{reduce_a}%")
with col2:
    st.metric("Пиковые дни", f"{экономия_peak:,.0f} ₽", delta=f"-{reduce_peak}%")
with col3:
    st.metric("Топ-магазин", f"{экономия_store:,.0f} ₽", delta=f"-{reduce_top_store}%")
with col4:
    st.metric("**Общая**", f"{общая_экономия:,.0f} ₽")

st.markdown("---")

# Конфиг графиков
plotly_config = {"toImageButtonOptions": {"format": "png", "filename": "график", "height": 600, "width": 1000, "scale": 2}}

# Табы
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Обзор", "📈 Графики", "⚠️ Аномалии и кластеры", "🔍 ABC и Pareto", "💡 Рекомендации"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Общие потери", f"{текущие_потери:,.0f} ₽")
    with col2:
        st.metric("Категорий", df['Категория'].nunique())
    with col3:
        st.metric("Магазинов", df['Магазин'].nunique())
    with col4:
        st.metric("Записей", len(df))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Потери по категориям")
        fig_cat = px.bar(суммарные_потери, x='Категория', y='СуммаПотерь', text='СуммаПотерь', color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_cat.update_traces(texttemplate='%{text:.0f} ₽', textposition='outside')
        st.plotly_chart(fig_cat, width='stretch', config=plotly_config)
    with col2:
        st.subheader("🏪 Потери по магазинам")
        fig_store = px.bar(потери_по_магазинам, x='Магазин', y='СуммаПотерь', text='СуммаПотерь', color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_store.update_traces(texttemplate='%{text:.0f} ₽', textposition='outside')
        st.plotly_chart(fig_store, width='stretch', config=plotly_config)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Динамика по месяцам")
        df_month = df.copy()
        df_month['Месяц'] = df_month['Дата'].dt.to_period('M').astype(str)
        monthly = df_month.groupby('Месяц')['СуммаПотерь'].sum().reset_index()
        fig_monthly = px.line(monthly, x='Месяц', y='СуммаПотерь', markers=True)
        st.plotly_chart(fig_monthly, width='stretch', config=plotly_config)
    with col2:
        st.subheader("🗓️ Динамика по кварталам")
        df_quarter = df.copy()
        df_quarter['Квартал'] = df_quarter['Дата'].dt.to_period('Q').astype(str)
        quarterly = df_quarter.groupby('Квартал')['СуммаПотерь'].sum().reset_index()
        fig_quarterly = px.line(quarterly, x='Квартал', y='СуммаПотерь', markers=True)
        st.plotly_chart(fig_quarterly, width='stretch', config=plotly_config)
    
    st.subheader("🌡️ Тепловая карта потерь")
    df_heat = df.copy()
    df_heat['День'] = df_heat['Дата'].dt.weekday.map(day_map)
    pivot = df_heat.pivot_table(values='СуммаПотерь', index='Категория', columns='День', aggfunc='sum', fill_value=0)
    pivot = pivot[['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']]
    fig_heat = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, color_continuous_scale='YlOrRd', text_auto=True)
    st.plotly_chart(fig_heat, width='stretch', config=plotly_config)
    
    st.markdown("---")
    st.subheader("📊 Средние потери по дням недели")
    day_avg = df_day.groupby('День')['СуммаПотерь'].mean().reindex(['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])
    fig_day_avg = px.bar(day_avg.reset_index(), x='День', y='СуммаПотерь', text='СуммаПотерь', color='СуммаПотерь', color_continuous_scale='Blues')
    fig_day_avg.update_traces(texttemplate='%{text:.0f} ₽', textposition='outside')
    st.plotly_chart(fig_day_avg, width='stretch', config=plotly_config)
    
    st.subheader("🔥 Топ-5 категорий в динамике")
    top5_cats = суммарные_потери.head(5)['Категория'].tolist()
    df_top5 = df[df['Категория'].isin(top5_cats)].copy()
    df_top5['Месяц'] = df_top5['Дата'].dt.to_period('M').astype(str)
    monthly_top5 = df_top5.groupby(['Месяц', 'Категория'])['СуммаПотерь'].sum().reset_index()
    fig_top5_dynamic = px.line(monthly_top5, x='Месяц', y='СуммаПотерь', color='Категория', markers=True)
    st.plotly_chart(fig_top5_dynamic, width='stretch', config=plotly_config)

with tab3:
    with st.spinner("Анализируем аномалии..."):
        df_anom = df.copy()
        df_anom['ДеньНедели'] = df_anom['Дата'].dt.weekday
        df_anom['Месяц'] = df_anom['Дата'].dt.month
        df_anom['ЛогПотерь'] = np.log1p(df_anom['СуммаПотерь'])
        
        top_cat = df_anom['Категория'].value_counts().head(10).index
        top_store = df_anom['Магазин'].value_counts().head(10).index
        df_anom['Категория_топ'] = df_anom['Категория'].where(df_anom['Категория'].isin(top_cat), 'Другие')
        df_anom['Магазин_топ'] = df_anom['Магазин'].where(df_anom['Магазин'].isin(top_store), 'Другие')
        
        features = pd.get_dummies(df_anom[['ЛогПотерь', 'ДеньНедели', 'Месяц', 'Категория_топ', 'Магазин_топ']])
        if len(features) >= 10:
            model = IsolationForest(contamination=0.05, random_state=42)
            df_anom['Аномалия'] = model.fit_predict(features)
            аномалии = df_anom[df_anom['Аномалия'] == -1]
            if len(аномалии) > 0:
                disp = аномалии[['Дата', 'Категория', 'СуммаПотерь', 'Магазин']].copy()
                disp['Дата'] = disp['Дата'].dt.strftime('%d.%m.%Y')
                with st.expander(f"📋 Аномалии ({len(аномалии)} шт.)"):
                    st.dataframe(disp, width='stretch')
                st.error(f"🚨 Выявлено {len(аномалии)} аномалий")
            else:
                st.success("✅ Аномалий не обнаружено")
        else:
            st.info("ℹ️ Недостаточно данных для анализа аномалий")
    
    with st.spinner("Кластеризация..."):
        if len(df) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['Кластер'] = kmeans.fit_predict(df[['СуммаПотерь']])
            cluster_means = df.groupby('Кластер')['СуммаПотерь'].mean().sort_values()
            labels = ['Низкий', 'Средний', 'Высокий']
            mapping = dict(zip(cluster_means.index, labels))
            df['Кластер'] = df['Кластер'].map(mapping)
            кластеры = df.groupby('Кластер')['СуммаПотерь'].describe().loc[labels].round(2)
            кластеры = кластеры.rename(columns={'count': 'Кол-во', 'mean': 'Среднее', 'min': 'Мин', '50%': 'Медиана', 'max': 'Макс'})
            st.subheader("🧩 Кластеры потерь")
            with st.expander("📋 Статистика кластеров"):
                st.dataframe(кластеры, width='stretch')

with tab4:
    st.subheader("📊 ABC-анализ категорий")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📋 Таблица ABC"):
            st.dataframe(abc[['Категория', 'СуммаПотерь', 'Доля_%', 'Накопительная_доля', 'ABC']], width='stretch')
    with col2:
        fig_abc = px.bar(abc, x='Категория', y='Накопительная_доля', color='ABC',
                         color_discrete_map={'A': '#ef4444', 'B': '#f59e0b', 'C': '#10b981'})
        fig_abc.add_hline(y=80, line_dash="dash", line_color="red")
        fig_abc.add_hline(y=95, line_dash="dash", line_color="orange")
        st.plotly_chart(fig_abc, width='stretch', config=plotly_config)
    
    st.markdown("---")
    st.subheader("🏪 Pareto-анализ магазинов")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📋 Таблица Pareto"):
            st.dataframe(pareto_store[['Магазин', 'СуммаПотерь', 'Доля_%', 'Накопительная_доля', 'Pareto']], width='stretch')
    with col2:
        fig_pareto = px.bar(pareto_store, x='Магазин', y='Накопительная_доля', color='Pareto',
                            color_discrete_map={'80%': '#ef4444', '95%': '#f59e0b', '100%': '#10b981'})
        fig_pareto.add_hline(y=80, line_dash="dash", line_color="red")
        st.plotly_chart(fig_pareto, width='stretch', config=plotly_config)
    
    st.markdown("---")
    st.subheader("📈 Прогноз по топ-3 категориям на 7 дней")
    with st.spinner("Обучаем Prophet..."):
        top3 = суммарные_потери.head(3)['Категория'].tolist()
        fig_multi = px.line(title="Прогноз по топ-3 категориям")
        tables = {}
        
        for cat in top3:
            daily = df[df['Категория'] == cat].groupby('Дата')['СуммаПотерь'].sum().reset_index()
            daily.columns = ['ds', 'y']
            if len(daily) >= 14:
                m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
                m.fit(daily)
                future = m.make_future_dataframe(periods=7)
                forecast = m.predict(future)
                
                fig_multi.add_scatter(x=daily['ds'], y=daily['y'], mode='lines+markers', name=f'{cat} (факт)')
                fig_multi.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name=f'{cat} (прогноз)', line=dict(dash='dash'))
                
                tbl = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).round(0)
                tbl['ds'] = tbl['ds'].dt.strftime('%d.%m.%Y')
                tables[cat] = tbl.rename(columns={'ds': 'Дата', 'yhat': 'Прогноз', 'yhat_lower': 'Мин', 'yhat_upper': 'Макс'})
        
        if tables:
            st.plotly_chart(fig_multi.update_layout(height=600), width='stretch', config=plotly_config)
            for cat, tbl in tables.items():
                with st.expander(f"📋 Прогноз для {cat}"):
                    st.dataframe(tbl, width='stretch')

with tab5:
    st.subheader("💡 Ранжированные рекомендации по экономии")
    
    рекомендации = [
        {"Текст": "Контроль A-класса категорий", "Экономия": экономия_a, "Снижение": reduce_a, "Цвет": "error"},
        {"Текст": "Аудит топ-магазина (Pareto 80%)", "Экономия": экономия_store, "Снижение": reduce_top_store, "Цвет": "error"},
        {"Текст": "Оптимизация пиковых дней недели", "Экономия": экономия_peak, "Снижение": reduce_peak, "Цвет": "warning"},
    ]
    рекомендации = sorted(рекомендации, key=lambda x: x["Экономия"], reverse=True)
    
    for i, r in enumerate(рекомендации, 1):
        text = f"{i}. {r['Текст']} — экономия до {r['Экономия']:,.0f} ₽ при снижении на {r['Снижение']}%"
        if r["Цвет"] == "error":
            st.error(text)
        elif r["Цвет"] == "warning":
            st.warning(text)
    
    st.success(f"🟢 **Максимальный потенциал:** {общая_экономия:,.0f} ₽ при реализации всех мер")

    st.markdown("---")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_export = df.copy()
        df_export['Дата'] = df_export['Дата'].dt.strftime('%d.%m.%Y')
        df_export.to_excel(writer, sheet_name='Данные', index=False)
        суммарные_потери.to_excel(writer, sheet_name='ПоКатегориям', index=False)
        потери_по_магазинам.to_excel(writer, sheet_name='ПоМагазинам', index=False)
        abc.to_excel(writer, sheet_name='ABC_категории', index=False)
        pareto_store.to_excel(writer, sheet_name='Pareto_магазины', index=False)
        pd.DataFrame([
            ["A-класс категорий", reduce_a, экономия_a],
            ["Пиковые дни", reduce_peak, экономия_peak],
            ["Топ-магазин", reduce_top_store, экономия_store],
            ["Общая экономия", "", общая_экономия]
        ], columns=['Сценарий', '% снижения', 'Экономия ₽']).to_excel(writer, sheet_name='What_if', index=False)
        pd.DataFrame([r["Текст"] + f" — {r['Экономия']:,.0f} ₽" for r in рекомендации], columns=['Рекомендация']).to_excel(writer, sheet_name='Рекомендации', index=False)
    buffer.seek(0)
    
    st.download_button(
        "📥 Скачать полный отчёт (Excel)",
        data=buffer.getvalue(),
        file_name=f"RetailLoss_Report_{datetime.today().strftime('%d%m%Y')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )