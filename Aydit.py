import os
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="RetailLoss Sentinel v8", layout="wide", page_icon="🛡️")

st.title("🛡️ RetailLoss Sentinel v8")
st.markdown("**AI-анализатор потерь ритейла** | Авто-распознавание колонок Excel")

with st.sidebar:
    st.header("📁 Загрузка данных")
    uploaded_file = st.file_uploader("Выберите Excel", type=["xlsx"])
    
    if st.button("🔄 Тестовые данные", type="primary"):
        st.session_state.use_test = True
        st.rerun()
    
    st.markdown("---")
    st.success("✅ scikit-learn + Prophet активны")
    st.info("Авто-распознаёт: Дата | Категория | СуммаПотерь | Магазин")

@st.cache_data
def сгенерировать_тестовые_данные():
    np.random.seed(42)
    сегодня = datetime(2025, 12, 20)
    даты = pd.date_range(end=сегодня, periods=180, freq='D')
    категории = np.random.choice(['Молочка', 'Мясо', 'Овощи', 'Алкоголь', 'Хлеб', 'Бакалея', 'Заморозка'], size=180)
    суммы_потерь = np.random.uniform(300, 7000, size=180).round(2)
    магазины = np.random.choice(['Магазин1', 'Магазин2', 'Магазин3', 'Магазин4', 'Магазин5'], size=180)
    df = pd.DataFrame({
        'Дата': даты,
        'Категория': категории,
        'СуммаПотерь': суммы_потерь,
        'Магазин': магазины
    })
    df['Дата'] = df['Дата'].dt.strftime('%d.%m.%Y')
    return df

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
            except: pass
    
    if not category_candidates:
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.1:
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
        st.error("❌ Не распознано. Используйте: Дата, Категория, СуммаПотерь, Магазин")
        st.stop()
    
    date_col = lower_columns[date_candidates[0]]
    category_col = lower_columns[category_candidates[0]]
    loss_col = lower_columns[loss_candidates[0]]
    store_col = lower_columns[store_candidates[0]]
    
    st.success(f"✅ Распознано: {date_col}→Дата | {category_col}→Категория | {loss_col}→СуммаПотерь | {store_col}→Магазин")
    
    df = df.rename(columns={
        date_col: 'Дата',
        category_col: 'Категория', 
        loss_col: 'СуммаПотерь',
        store_col: 'Магазин'
    })
    return df

def выполнить_анализ(df_original):
    df = df_original.copy()
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
    
    if df['Дата'].isnull().any():
        st.error("❌ Ошибка в датах")
        st.stop()
    
    df = df.sort_values('Дата').reset_index(drop=True)
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔧 Фильтры")
        магазины_список = ['Все'] + sorted(df['Магазин'].astype(str).unique().tolist())
        выбранные_магазины = st.multiselect("Магазины", магазины_список, default='Все')
        
        категории_список = ['Все'] + sorted(df['Категория'].astype(str).unique().tolist())
        выбранные_категории = st.multiselect("Категории", категории_список, default='Все')
        
        min_date = df['Дата'].min().date()
        max_date = df['Дата'].max().date()
        выбранный_период = st.date_input("Период", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    if 'Все' not in выбранные_магазины:
        df = df[df['Магазин'].isin(выбранные_магазины)]
    if 'Все' not in выбранные_категории:
        df = df[df['Категория'].isin(выбранные_категории)]
    
    start_date, end_date = выбранный_период
    df = df[(df['Дата'].dt.date >= start_date) & (df['Дата'].dt.date <= end_date)]
    
    if df.empty:
        st.warning("⚠️ Нет данных по фильтрам")
        st.stop()
    
    длина_периода = (end_date - start_date).days + 1
    prev_start = start_date - timedelta(days=длина_периода)
    prev_end = start_date - timedelta(days=1)
    
    текущие_потери = df['СуммаПотерь'].sum()
    df_prev = df_original.copy()
    df_prev['Дата'] = pd.to_datetime(df_prev['Дата'], dayfirst=True)
    df_prev = df_prev[(df_prev['Дата'].dt.date >= prev_start) & (df_prev['Дата'].dt.date <= prev_end)]
    предыдущие_потери = df_prev['СуммаПотерь'].sum()
    изменение = ((текущие_потери - предыдущие_потери) / предыдущие_потери * 100) if предыдущие_потери > 0 else 0
    
    df['Индекс'] = np.arange(len(df))
    модель_anom = IsolationForest(contamination=0.1, random_state=42)
    df['Аномалия'] = модель_anom.fit_predict(df[['Индекс', 'СуммаПотерь']])
    аномалии = df[df['Аномалия'] == -1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Общие потери", f"{текущие_потери:,.0f} ₽", delta=f"{изменение:+.1f}%")
    with col2:
        st.metric("Категорий", df['Категория'].nunique())
    with col3:
        st.metric("Магазинов", df['Магазин'].nunique())
    with col4:
        st.metric("Аномалий", len(аномалии))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 По категориям")
        суммарные_потери = df.groupby('Категория')['СуммаПотерь'].sum().reset_index().sort_values('СуммаПотерь', ascending=False)
        fig_cat = px.bar(суммарные_потери, x='Категория', y='СуммаПотерь', text='СуммаПотерь', color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_cat.update_traces(texttemplate='%{text:.0f} ₽', textposition='outside')
        fig_cat.update_layout(height=500)
        st.plotly_chart(fig_cat, width='stretch')
    
    with col2:
        st.subheader("🏪 По магазинам")
        потери_по_магазинам = df.groupby('Магазин')['СуммаПотерь'].sum().reset_index().sort_values('СуммаПотерь', ascending=False)
        fig_store = px.bar(потери_по_магазинам, x='Магазин', y='СуммаПотерь', text='СуммаПотерь', color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_store.update_traces(texttemplate='%{text:.0f} ₽', textposition='outside')
        fig_store.update_layout(height=500)
        st.plotly_chart(fig_store, width='stretch')
    
    st.subheader("📊 Динамика")
    df_month = df.copy()
    df_month['Месяц'] = df_month['Дата'].dt.to_period('M').astype(str)
    df_quarter = df.copy()
    df_quarter['Квартал'] = df_quarter['Дата'].dt.to_period('Q').astype(str)
    
    monthly_losses = df_month.groupby('Месяц')['СуммаПотерь'].sum().reset_index()
    quarterly_losses = df_quarter.groupby('Квартал')['СуммаПотерь'].sum().reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_monthly = px.line(monthly_losses, x='Месяц', y='СуммаПотерь', markers=True, title='Месяцы')
        fig_monthly.update_layout(height=400)
        st.plotly_chart(fig_monthly, width='stretch')
    
    with col2:
        fig_quarterly = px.line(quarterly_losses, x='Квартал', y='СуммаПотерь', markers=True, title='Кварталы')
        fig_quarterly.update_layout(height=400)
        st.plotly_chart(fig_quarterly, width='stretch')
    
    st.subheader("🌡️ Тепловая карта")
    df_heat = df.copy()
    df_heat['ДеньНедели'] = df_heat['Дата'].dt.day_name(locale='ru_RU')
    pivot_heat = df_heat.pivot_table(values='СуммаПотерь', index='Категория', columns='ДеньНедели', aggfunc='sum', fill_value=0)
    дни_порядок = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    pivot_heat = pivot_heat.reindex(columns=дни_порядок)
    
    fig_heat = px.imshow(pivot_heat.values, x=дни_порядок, y=pivot_heat.index, color_continuous_scale='YlOrRd', text_auto=True, aspect="auto")
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, width='stretch')
    
    if len(аномалии) > 0:
        st.subheader("⚠️ Аномалии")
        аномалии_disp = аномалии.copy()
        аномалии_disp['Дата'] = аномалии_disp['Дата'].dt.strftime('%d.%m.%Y')
        st.dataframe(аномалии_disp[['Дата', 'Категория', 'СуммаПотерь', 'Магазин']], width='stretch')
    else:
        st.success("✅ Аномалий нет")
    
    if len(df) >= 3:
        X_cluster = df[['СуммаПотерь']].values
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Кластер'] = kmeans.fit_predict(X_cluster)
        
        cluster_means = df.groupby('Кластер')['СуммаПотерь'].mean().sort_values()
        labels = ['Низкий', 'Средний', 'Высокий']
        mapping = dict(zip(cluster_means.index, labels))
        df['Кластер'] = df['Кластер'].map(mapping)
        кластеры = df.groupby('Кластер')['СуммаПотерь'].describe().loc[labels]
        кластеры = кластеры.rename(columns={
            'count': 'Количество', 'mean': 'Среднее', 'std': 'Стд. отклонение',
            'min': 'Мин', '25%': '25%', '50%': 'Медиана', '75%': '75%', 'max': 'Макс'
        }).round(2)
        st.subheader("🧩 Кластеры")
        st.dataframe(кластеры, width='stretch')
    
    ежедневные_потери = df.groupby('Дата')['СуммаПотерь'].sum().reset_index()
    ежедневные_потери.columns = ['ds', 'y']
    
    if len(ежедневные_потери) >= 14:
        st.subheader("📈 Прогноз (Prophet)")
        модель_prophet = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        модель_prophet.fit(ежедневные_потери)
        future = модель_prophet.make_future_dataframe(periods=7)
        forecast = модель_prophet.predict(future)
        прогноз_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).rename(columns={'ds': 'Дата', 'yhat': 'Прогноз'}).round(2)
        
        fig_prog = px.line(ежедневные_потери, x='ds', y='y', title='Прогноз потерь')
        fig_prog.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз', line=dict(color='#f87171', dash='dash'))
        fig_prog.update_layout(height=600)
        st.plotly_chart(fig_prog, width='stretch')
        
        прогноз_disp = прогноз_df.copy()
        прогноз_disp['Дата'] = прогноз_disp['Дата'].dt.strftime('%d.%m.%Y')
        прогноз_disp = прогноз_disp[['Дата', 'Прогноз', 'yhat_lower', 'yhat_upper']].rename(columns={'yhat_lower': 'Мин', 'yhat_upper': 'Макс'})
        st.dataframe(прогноз_disp, width='stretch')
    else:
        st.warning("⚠️ Для прогноза нужно ≥14 дней")
    
    top_category = суммарные_потери.iloc[0]['Категория'] if not суммарные_потери.empty else "нет"
    top_store = потери_по_магазинам.iloc[0]['Магазин'] if not потери_по_магазинам.empty else "нет"
    
    st.subheader("💡 Рекомендации")
    st.markdown(f"""
    🔴 **Категория:** {top_category} — усилить контроль  
    🔴 **Магазин:** {top_store} — провести аудит  
    💰 **Экономия:** 20-30% от {текущие_потери:,.0f}₽
    """)
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_export = df.copy()
        df_export['Дата'] = df_export['Дата'].dt.strftime('%d.%m.%Y')
        df_export.to_excel(writer, sheet_name='Данные', index=False)
        суммарные_потери.to_excel(writer, sheet_name='Категории')
        потери_по_магазинам.to_excel(writer, sheet_name='Магазины')
        if len(аномалии) > 0:
            аномалии_exp = аномалии.copy()
            аномалии_exp['Дата'] = аномалии_exp['Дата'].dt.strftime('%d.%m.%Y')
            аномалии_exp.to_excel(writer, sheet_name='Аномалии', index=False)
        monthly_losses.to_excel(writer, sheet_name='Месяц')
        pivot_heat.to_excel(writer, sheet_name='Тепло')
        if 'кластеры' in locals():
            кластеры.to_excel(writer, sheet_name='Кластеры')
    
    buffer.seek(0)
    st.download_button(
        "📥 Полный отчёт Excel",
        buffer,
        f"отчет_{datetime.now().strftime('%d%m%Y_%H%M')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(uploaded_file, sheet_name=xls.sheet_names[0], engine='openpyxl')
        df = detect_columns(df)
        выполнить_анализ(df)
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
elif st.session_state.get('use_test', False):
    df = сгенерировать_тестовые_данные()
    df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
    выполнить_анализ(df)
    st.session_state.use_test = False
else:
    st.info("👆 Загрузите Excel или 'Тестовые данные'")
    st.dataframe(сгенерировать_тестовые_данные().head(10), width='stretch')
