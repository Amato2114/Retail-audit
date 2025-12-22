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

st.set_page_config(page_title="🛡️ RetailLoss Sentinel v8", layout="wide", page_icon="🛡️")

st.title("🛡️ RetailLoss Sentinel v8")
st.markdown("**AI-анализатор потерь ритейла** | 🚀 Авто-распознавание колонок Excel")

with st.sidebar:
    st.header("📁 Загрузка данных")
    uploaded_file = st.file_uploader("📄 Excel файл", type=["xlsx"])
    
    if st.button("🔄 Генерировать тестовые данные", type="primary", use_container_width=False):
        st.session_state.use_test = True
        st.rerun()
    
    st.markdown("---")
    st.success("✅ Готово: Prophet + scikit-learn + авто-детект колонок")
    st.info("**Авто-распознаёт:** Дата | Категория | СуммаПотерь | Магазин")

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
    
    # Поиск по ключевым словам
    date_candidates = [col for col in lower_columns if any(kw in col for kw in ['дат', 'date'])]
    category_candidates = [col for col in lower_columns if any(kw in col for kw in ['кат', 'cat', 'товар', 'продукт'])]
    loss_candidates = [col for col in lower_columns if any(kw in col for kw in ['пот', 'loss', 'сумм', 'убыт', 'спис'])]
    store_candidates = [col for col in lower_columns if any(kw in col for kw in ['маг', 'store', 'филиал', 'торг'])]
    
    # Fallback по типам
    if not date_candidates:
        for col in df.columns:
            try:
                if pd.to_datetime(df[col], errors='coerce', dayfirst=True).notna().mean() > 0.7:
                    date_candidates = [col.lower()]
                    break
            except: pass
    
    if not category_candidates:
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.1:
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
            if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.05:
                store_candidates = [col.lower()]
                break
    
    if not all([date_candidates, category_candidates, loss_candidates, store_candidates]):
        st.error("❌ Не удалось распознать колонки. Используйте: **Дата**, **Категория**, **СуммаПотерь**, **Магазин**")
        st.stop()
    
    cols = [lower_columns[candidates[0]] for candidates in [date_candidates, category_candidates, loss_candidates, store_candidates]]
    st.success(f"✅ Распознано: {', '.join(cols)}")
    return df.rename(columns=dict(zip(cols, ['Дата', 'Категория', 'СуммаПотерь', 'Магазин'])))

def выполнить_анализ(df_original):
    df = df_original.copy()
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
    if df['Дата'].isnull().any(): 
        st.error("❌ Ошибка в датах"); 
        st.stop()
    
    df = df.sort_values('Дата').reset_index(drop=True)
    
    # Фильтры
    with st.sidebar:
        st.subheader("🔧 Фильтры")
        col1, col2 = st.columns(2)
        with col1:
            stores = ['Все'] + sorted(df['Магазин'].unique())
            selected_stores = st.multiselect("Магазины", stores, default='Все')
        with col2:
            cats = ['Все'] + sorted(df['Категория'].unique())
            selected_cats = st.multiselect("Категории", cats, default='Все')
        period = st.date_input("Период", (df['Дата'].min().date(), df['Дата'].max().date()))
    
    # Фильтрация
    df_filt = df.copy()
    if 'Все' not in selected_stores: df_filt = df_filt[df_filt['Магазин'].isin(selected_stores)]
    if 'Все' not in selected_cats: df_filt = df_filt[df_filt['Категория'].isin(selected_cats)]
    df_filt = df_filt[(df_filt['Дата'].dt.date >= period[0]) & (df_filt['Дата'].dt.date <= period[1])]
    
    if df_filt.empty: 
        st.warning("⚠️ Нет данных по фильтрам")
        st.stop()
    
    # Метрики
    total_loss = df_filt['СуммаПотерь'].sum()
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("💰 Общие потери", f"{total_loss:,.0f} ₽")
    with col2: st.metric("📦 Категорий", df_filt['Категория'].nunique())
    with col3: st.metric("🏪 Магазинов", df_filt['Магазин'].nunique())
    
    # Аномалии
    df_filt['Индекс'] = range(len(df_filt))
    iso = IsolationForest(contamination=0.1, random_state=42)
    df_filt['Аномалия'] = iso.fit_predict(df_filt[['Индекс', 'СуммаПотерь']])
    anomalies = df_filt[df_filt['Аномалия'] == -1]
    with col4: st.metric("🚨 Аномалий", len(anomalies))
    
    st.markdown("---")
    
    # Графики
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Потери по категориям")
        cat_agg = df_filt.groupby('Категория')['СуммаПотерь'].sum().sort_values(ascending=False).reset_index()
        fig_cat = px.bar(cat_agg, x='Категория', y='СуммаПотерь', text='СуммаПотерь',
                        color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_cat.update_traces(texttemplate='%{text:,.0f} ₽', textposition='outside')
        fig_cat.update_layout(height=450)
        st.plotly_chart(fig_cat, width='stretch')
    
    with col2:
        st.subheader("🏪 Потери по магазинам")
        store_agg = df_filt.groupby('Магазин')['СуммаПотерь'].sum().sort_values(ascending=False).reset_index()
        fig_store = px.bar(store_agg, x='Магазин', y='СуммаПотерь', text='СуммаПотерь',
                          color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_store.update_traces(texttemplate='%{text:,.0f} ₽', textposition='outside')
        fig_store.update_layout(height=450)
        st.plotly_chart(fig_store, width='stretch')
    
    # Динамика
    st.subheader("📈 Динамика по месяцам")
    df_month = df_filt.copy()
    df_month['Месяц'] = df_month['Дата'].dt.to_period('M').astype(str)
    monthly = df_month.groupby('Месяц')['СуммаПотерь'].sum().reset_index()
    fig_month = px.line(monthly, x='Месяц', y='СуммаПотерь', markers=True, 
                       title="Потери по месяцам")
    fig_month.update_layout(height=450)
    st.plotly_chart(fig_month, width='stretch')
    
    # Тепловая карта
    st.subheader("🌡️ Тепловая карта (Категории × День недели)")
    df_heat = df_filt.copy()
    df_heat['День'] = df_heat['Дата'].dt.day_name(locale='ru_RU')
    pivot = df_heat.pivot_table('СуммаПотерь', 'Категория', 'День', 'sum', fill_value=0)
    days_order = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    pivot = pivot.reindex(columns=days_order)
    
    fig_heat = px.imshow(pivot, color_continuous_scale='YlOrRd', text_auto=True, aspect="auto")
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, width='stretch')
    
    # Аномалии таблица
    if len(anomalies) > 0:
        st.subheader("⚠️ Аномалии (Isolation Forest)")
        display_anom = anomalies[['Дата', 'Категория', 'СуммаПотерь', 'Магазин']].copy()
        display_anom['Дата'] = display_anom['Дата'].dt.strftime('%d.%m.%Y')
        st.dataframe(display_anom, width='stretch')
    else:
        st.success("✅ Аномалий не найдено")
    
    # Кластеризация
    if len(df_filt) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_filt['Кластер'] = kmeans.fit_predict(df_filt[['СуммаПотерь']])
        clusters = df_filt.groupby('Кластер')['СуммаПотерь'].describe().round(0)
        clusters.columns = ['Кол-во', 'Среднее', 'Стд', 'Мин', '25%', '50%', '75%', 'Макс']
        st.subheader("🧩 Кластеры потерь (K-Means)")
        st.dataframe(clusters)
    
    # Prophet прогноз
    if len(df_filt) >= 14:
        st.subheader("🔮 Прогноз на 7 дней (Prophet)")
        daily = df_filt.groupby('Дата')['СуммаПотерь'].sum().reset_index()
        daily.columns = ['ds', 'y']
        
        m = Prophet(daily_seasonality=True, weekly_seasonality=True)
        m.fit(daily)
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        
        fig_prophet = m.plot(forecast)
        st.pyplot(fig_prophet)
        
        fc_future = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
        fc_future.columns = ['Дата', 'Прогноз', 'Мин', 'Макс']
        fc_future['Дата'] = fc_future['Дата'].dt.strftime('%d.%m.%Y')
        st.dataframe(fc_future.round(0))
    
    # Excel экспорт
    st.subheader("📥 Скачать отчет")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_filt.to_excel(writer, 'Данные', index=False)
        cat_agg.to_excel(writer, 'Категории')
        store_agg.to_excel(writer, 'Магазины')
        monthly.to_excel(writer, 'Месяц')
        if len(anomalies) > 0: anomalies.to_excel(writer, 'Аномалии', index=False)
        pivot.to_excel(writer, 'Тепловая_карта')
    
    buffer.seek(0)
    st.download_button(
        "📊 Полный отчет Excel",
        buffer,
        f"RetailLoss_отчет_{datetime.now().strftime('%d%m%Y_%H%M')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Основной поток
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df = detect_columns(df)
        выполнить_анализ(df)
    except Exception as e:
        st.error(f"❌ Ошибка файла: {e}")
        st.info("💡 Проверьте формат Excel")
elif st.session_state.get('use_test', False):
    df = сгенерировать_тестовые_данные()
    df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
    выполнить_анализ(df)
    st.session_state.use_test = False
else:
    st.info("👆 **Загрузите Excel** или нажмите **'Генерировать тестовые данные'**")
    st.markdown("### 📋 Пример данных")
    st.dataframe(сгенерировать_тестовые_данные().head(10), width='stretch')
