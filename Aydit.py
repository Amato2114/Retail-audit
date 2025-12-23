import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import io

# Настройка страницы
st.set_page_config(page_title="RetailLoss Sentinel v8", layout="wide", page_icon="🛡️")

# Заголовок и описание
st.title("🛡️ RetailLoss Sentinel v8")
st.markdown("**Инновационный AI-анализатор потерь для гипермаркетов**")
st.markdown("Авто-распознавание колонок • Улучшенный ML • ABC-анализ • Прогноз по категориям")

# Сайдбар
with st.sidebar:
    st.header("📁 Загрузка данных")
    uploaded_file = st.file_uploader("Выберите файл Excel", type=["xlsx"])
    
    if st.button("🔄 Тестовые данные (300 строк)"):
        st.session_state.use_test = True
        st.rerun()
    
    st.markdown("---")
    st.success("✅ Все библиотеки готовы!")
    st.info("Авто-распознавание колонок по ключевым словам и типам данных.")

# Кэшированные тестовые данные (300 строк)
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

# Фильтры в сайдбаре
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

# Сравнение с предыдущим периодом
длина_периода = (выбранный_период[1] - выбранный_период[0]).days + 1
prev_start = выбранный_период[0] - timedelta(days=длина_периода)
prev_end = выбранный_период[0] - timedelta(days=1)
df_prev = df_raw[(df_raw['Дата'].dt.date >= prev_start) & (df_raw['Дата'].dt.date <= prev_end)]
изменение = ((df['СуммаПотерь'].sum() - df_prev['СуммаПотерь'].sum()) / df_prev['СуммаПотерь'].sum() * 100) if len(df_prev) > 0 and df_prev['СуммаПотерь'].sum() > 0 else 0

текущие_потери = df['СуммаПотерь'].sum()

# Большая центральная метрика
st.markdown(f"""
    <div style='text-align: center; margin: 30px 0;'>
        <h1 style='color: #ef4444; font-size: 48px; margin: 0;'>
            {текущие_потери:,.0f} ₽
        </h1>
        <p style='font-size: 20px; color: gray; margin: 5px 0;'>
            Общие потери за выбранный период
        </p>
        <p style='font-size: 18px; color: {"#ef4444" if изменение > 0 else "#10b981"}; margin: 0;'>
            {изменение:+.1f}% к предыдущему периоду
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Конфиг для встроенной кнопки скачивания PNG в Plotly
plotly_config = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": "график_потерь",
        "height": 600,
        "width": 1000,
        "scale": 2
    }
}

# Табы
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Обзор", "📈 Графики", "⚠️ Аномалии и кластеры", "🔍 ABC и прогнозы", "💡 Рекомендации"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Общие потери", f"{текущие_потери:,.0f} ₽", delta=f"{изменение:+.1f}%")
    with col2:
        st.metric("Категорий", df['Категория'].nunique())
    with col3:
        st.metric("Магазинов", df['Магазин'].nunique())
    with col4:
        st.metric("Записей", len(df))
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Потери по категориям")
        суммарные_потери = df.groupby('Категория')['СуммаПотерь'].sum().reset_index().sort_values('СуммаПотерь', ascending=False)
        fig_cat = px.bar(суммарные_потери, x='Категория', y='СуммаПотерь', text='СуммаПотерь', color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_cat.update_traces(texttemplate='%{text:.0f} ₽', textposition='outside')
        fig_cat.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True, config=plotly_config)
    with col2:
        st.subheader("🏪 Потери по магазинам")
        потери_по_магазинам = df.groupby('Магазин')['СуммаПотерь'].sum().reset_index().sort_values('СуммаПотерь', ascending=False)
        fig_store = px.bar(потери_по_магазинам, x='Магазин', y='СуммаПотерь', text='СуммаПотерь', color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_store.update_traces(texttemplate='%{text:.0f} ₽', textposition='outside')
        fig_store.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_store, use_container_width=True, config=plotly_config)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Динамика по месяцам")
        df_month = df.copy()
        df_month['Месяц'] = df_month['Дата'].dt.to_period('M').astype(str)
        monthly = df_month.groupby('Месяц')['СуммаПотерь'].sum().reset_index()
        fig_monthly = px.line(monthly, x='Месяц', y='СуммаПотерь', markers=True)
        fig_monthly.update_layout(height=450)
        st.plotly_chart(fig_monthly, use_container_width=True, config=plotly_config)
    with col2:
        st.subheader("🗓️ Динамика по кварталам")
        df_quarter = df.copy()
        df_quarter['Квартал'] = df_quarter['Дата'].dt.to_period('Q').astype(str)
        quarterly = df_quarter.groupby('Квартал')['СуммаПотерь'].sum().reset_index()
        fig_quarterly = px.line(quarterly, x='Квартал', y='СуммаПотерь', markers=True)
        fig_quarterly.update_layout(height=450)
        st.plotly_chart(fig_quarterly, use_container_width=True, config=plotly_config)
    
    st.subheader("🌡️ Тепловая карта потерь (категории × дни недели)")
    df_heat = df.copy()
    day_map = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}
    df_heat['День'] = df_heat['Дата'].dt.weekday.map(day_map)
    pivot = df_heat.pivot_table(values='СуммаПотерь', index='Категория', columns='День', aggfunc='sum', fill_value=0)
    pivot = pivot[['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']]
    fig_heat = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, color_continuous_scale='YlOrRd', text_auto=True, aspect="auto")
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True, config=plotly_config)

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
                st.dataframe(disp, use_container_width=True)
                st.error(f"🚨 Выявлено {len(аномалии)} аномалий")
            else:
                st.success("✅ Аномалий не обнаружено")
        else:
            st.info("ℹ️ Недостаточно данных для анализа аномалий")
    
    with st.spinner("Выполняем кластеризацию..."):
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
            st.dataframe(кластеры, use_container_width=True)

with tab4:
    st.subheader("📊 ABC-анализ категорий")
    abc = суммарные_потери.copy()
    abc['Доля_%'] = (abc['СуммаПотерь'] / abc['СуммаПотерь'].sum() * 100).round(2)
    abc['Накопительная_доля'] = abc['Доля_%'].cumsum()
    abc['ABC'] = abc['Накопительная_доля'].apply(lambda x: 'A' if x <= 80 else 'B' if x <= 95 else 'C')
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(abc[['Категория', 'СуммаПотерь', 'Доля_%', 'Накопительная_доля', 'ABC']], use_container_width=True)
    with col2:
        fig_abc = px.bar(abc, x='Категория', y='Накопительная_доля', color='ABC',
                         color_discrete_map={'A': '#ef4444', 'B': '#f59e0b', 'C': '#10b981'})
        fig_abc.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
        fig_abc.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="95%")
        st.plotly_chart(fig_abc, use_container_width=True, config=plotly_config)
    
    st.info("**A-класс** — 80% потерь. **B** — следующий 15%. **C** — остальное.")

    st.markdown("---")
    st.subheader("📈 Прогноз по топ-3 категориям на 7 дней")
    with st.spinner("Обучаем модели Prophet..."):
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
            st.plotly_chart(fig_multi.update_layout(height=600), use_container_width=True, config=plotly_config)
            for cat, tbl in tables.items():
                st.markdown(f"**{cat}**")
                st.dataframe(tbl, use_container_width=True)

with tab5:
    st.subheader("💡 Персонализированные рекомендации")
    top_cat = суммарные_потери.iloc[0]['Категория'] if len(суммарные_потери) > 0 else "—"
    top_store = потери_по_магазинам.iloc[0]['Магазин'] if len(потери_по_магазинам) > 0 else "—"
    peak_day = pivot.sum().idxmax() if 'pivot' in locals() else "—"
    
    рекомендации = [
        f"🔴 **Высокий приоритет:** Усилить контроль в категории «{top_cat}» — лидер по потерям.",
        f"🔴 **Высокий приоритет:** Провести аудит магазина «{top_store}» — максимальные потери.",
        f"🟡 **Пик потерь:** В день «{peak_day}» — добавить проверки полок и приёмки.",
        f"🟢 **ABC-анализ:** 80% потерь в A-классе — фокус здесь даст максимальную экономию.",
        f"💡 **Прогноз:** Следите за ростом в топ-категориях на следующей неделе.",
        f"💰 **Потенциальная экономия:** 20–30% при внедрении мер контроля."
    ]
    
    for r in рекомендации:
        st.markdown(f"• {r}")
    
    st.markdown("---")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_export = df.copy()
        df_export['Дата'] = df_export['Дата'].dt.strftime('%d.%m.%Y')
        df_export.to_excel(writer, sheet_name='Данные', index=False)
        суммарные_потери.to_excel(writer, sheet_name='ПоКатегориям', index=False)
        потери_по_магазинам.to_excel(writer, sheet_name='ПоМагазинам', index=False)
        abc.to_excel(writer, sheet_name='ABC_анализ', index=False)
        pd.DataFrame(рекомендации, columns=['Рекомендация']).to_excel(writer, sheet_name='Рекомендации', index=False)
    buffer.seek(0)
    
    st.download_button(
        "📥 Скачать полный отчёт (Excel)",
        data=buffer,
        file_name=f"RetailLoss_Report_{datetime.today().strftime('%d%m%Y')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )