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
st.set_page_config(page_title="RetailLoss Sentinel v8 (Auto Columns)", layout="wide", page_icon="🛡️")

# Заголовок
st.title("🛡️ RetailLoss Sentinel v8")
st.markdown("**Инновационный AI-анализатор потерь для гипермаркетов**")
st.markdown("**Новый фича:** Авто-распознавание колонок в Excel (по именам/типам). Не нужно строгая форма — угадывает 'Дата', 'Категория', 'СуммаПотерь', 'Магазин'. Все фичи реализованы полностью.")

# Сайдбар
with st.sidebar:
    st.header("📁 Загрузка данных")
    uploaded_file = st.file_uploader("Выберите файл Excel", type=["xlsx"])
    
    if st.button("🔄 Тестовые данные"):
        st.session_state.use_test = True
        st.rerun()
    
    st.markdown("---")
    st.success("✅ scikit-learn активен — ML полностью готов!")
    st.info("""
    **Авто-распознавание колонок:** 
    - Дата: 'date', 'дата', datetime-like.
    - Категория: 'category', 'категория', string-like.
    - СуммаПотерь: 'loss', 'потери', 'sum', float-like >0.
    - Магазин: 'store', 'магазин', string-like.
    Если не угадало — ошибка с подсказкой.
    """)

# Тестовые данные
@st.cache_data
def сгенерировать_тестовые_данные():
    np.random.seed(42)
    сегодня = datetime.now()  # Динамическая дата
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
    
    if not date_candidates or not category_candidates or not loss_candidates or not store_candidates:
        st.error("❌ Не удалось автоматически распознать все обязательные колонки. Проверьте файл или переименуйте колонки ближе к 'Дата', 'Категория', 'СуммаПотерь', 'Магазин'.")
        st.stop()
    
    date_col = lower_columns[date_candidates[0]]
    category_col = lower_columns[category_candidates[0]]
    loss_col = lower_columns[loss_candidates[0]]
    store_col = lower_columns[store_candidates[0]]
    
    st.success(f"✅ Автоматически распознаны колонки:\n- Дата: **{date_col}**\n- Категория: **{category_col}**\n- СуммаПотерь: **{loss_col}**\n- Магазин: **{store_col}**")
    
    df = df.rename(columns={
        date_col: 'Дата',
        category_col: 'Категория',
        loss_col: 'СуммаПотерь',
        store_col: 'Магазин'
    })
    
    return df

# Основная функция анализа
def выполнить_анализ(df_original):
    df = df_original.copy()
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
    
    if df['Дата'].isnull().any():
        st.error("❌ Ошибка: Некоторые даты не распознаны.")
        st.stop()
    
    df = df.sort_values('Дата').reset_index(drop=True)
    
    # Фильтры
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔧 Фильтры")
        
        магазины_список = ['Все'] + sorted(df['Магазин'].astype(str).unique().tolist())
        выбранные_магазины = st.multiselect("Магазины", магазины_список, default='Все')
        
        категории_список = ['Все'] + sorted(df['Категория'].astype(str).unique().tolist())
        выбранные_категории = st.multiselect("Категории", категории_список, default='Все')
        
        min_date = df['Дата'].min().date()
        max_date = df['Дата'].max().date()
        выбранный_период = st.date_input("Период дат", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    # Применение фильтров
    if 'Все' not in выбранные_магазины:
        df = df[df['Магазин'].isin(выбранные_магазины)]
    if 'Все' not in выбранные_категории:
        df = df[df['Категория'].isin(выбранные_категории)]
    
    start_date, end_date = выбранный_период
    df = df[(df['Дата'].dt.date >= start_date) & (df['Дата'].dt.date <= end_date)]
    
    if df.empty:
        st.warning("⚠️ Нет данных по выбранным фильтрам.")
        st.stop()
    
    # Сравнение периодов
    длина_периода = (end_date - start_date).days + 1
    prev_start = start_date - timedelta(days=длина_периода)
    prev_end = start_date - timedelta(days=1)
    
    текущие_потери = df['СуммаПотерь'].sum()
    df_prev = df_original.copy()
    df_prev['Дата'] = pd.to_datetime(df_prev['Дата'], dayfirst=True)
    df_prev = df_prev[(df_prev['Дата'].dt.date >= prev_start) & (df_prev['Дата'].dt.date <= prev_end)]
    предыдущие_потери = df_prev['СуммаПотерь'].sum()
    изменение = ((текущие_потери - предыдущие_потери) / предыдущие_потери * 100) if предыдущие_потери > 0 else 0
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Общие потери", f"{текущие_потери:,.0f} ₽", delta=f"{изменение:+.1f}% vs прошлый")
    with col2:
        st.metric("Категорий", df['Категория'].nunique())
    with col3:
        st.metric("Магазинов", df['Магазин'].nunique())
    with col4:
        st.metric("Записей", len(df))
    
    st.markdown("---")
    
    # Графики потерь
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Потери по категориям")
        суммарные_потери = df.groupby('Категория')['СуммаПотерь'].sum().reset_index().sort_values('СуммаПотерь', ascending=False)
        fig_cat = px.bar(суммарные_потери, x='Категория', y='СуммаПотерь', text='СуммаПотерь', color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_cat.update_traces(texttemplate='%{text:.0f} ₽', textposition='outside')
        fig_cat.update_layout(yaxis_title='Сумма потерь, ₽', height=500)
        st.plotly_chart(fig_cat, width='stretch')
    with col2:
        st.subheader("🏪 Потери по магазинам")
        потери_по_магазинам = df.groupby('Магазин')['СуммаПотерь'].sum().reset_index().sort_values('СуммаПотерь', ascending=False)
        fig_store = px.bar(потери_по_магазинам, x='Магазин', y='СуммаПотерь', text='СуммаПотерь', color='СуммаПотерь', color_continuous_scale='YlOrRd')
        fig_store.update_traces(texttemplate='%{text:.0f} ₽', textposition='outside')
        fig_store.update_layout(yaxis_title='Сумма потерь, ₽', height=500)
        st.plotly_chart(fig_store, width='stretch')
    
    # Динамика по месяцам и кварталам
    st.subheader("📊 Динамика потерь по месяцам и кварталам")
    df_month = df.copy()
    df_month['Месяц'] = df_month['Дата'].dt.to_period('M').astype(str)
    df_quarter = df.copy()
    df_quarter['Квартал'] = df_quarter['Дата'].dt.to_period('Q').astype(str)
    
    monthly_losses = df_month.groupby('Месяц')['СуммаПотерь'].sum().reset_index()
    quarterly_losses = df_quarter.groupby('Квартал')['СуммаПотерь'].sum().reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_monthly = px.line(monthly_losses, x='Месяц', y='СуммаПотерь', markers=True, title='По месяцам')
        fig_monthly.update_layout(yaxis_title='Сумма потерь, ₽', height=400)
        st.plotly_chart(fig_monthly, width='stretch')
    with col2:
        fig_quarterly = px.line(quarterly_losses, x='Квартал', y='СуммаПотерь', markers=True, title='По кварталам')
        fig_quarterly.update_layout(yaxis_title='Сумма потерь, ₽', height=400)
        st.plotly_chart(fig_quarterly, width='stretch')
    
    buf_month = io.BytesIO()
    try:
        fig_monthly.write_image(buf_month, format='png')
        buf_month.seek(0)
        st.download_button("📥 Скачать график по месяцам (PNG)", buf_month, file_name="динамика_по_месяцам.png", mime="image/png")
    except Exception as e:
        st.warning(f"⚠️ Экспорт PNG недоступен: {str(e)}")
    
    # Тепловая карта
    st.subheader("🌡️ Тепловая карта потерь (категории × дни недели)")
    df_heat = df.copy()
    day_map = {0: 'Понедельник', 1: 'Вторник', 2: 'Среда', 3: 'Четверг', 4: 'Пятница', 5: 'Суббота', 6: 'Воскресенье'}
    df_heat['ДеньНедели'] = df_heat['Дата'].dt.weekday.map(day_map)
    pivot_heat = df_heat.pivot_table(values='СуммаПотерь', index='Категория', columns='ДеньНедели', aggfunc='sum', fill_value=0)
    дни_порядок = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    pivot_heat = pivot_heat.reindex(columns=дни_порядок)
    
    fig_heat = px.imshow(pivot_heat.values, x=дни_порядок, y=pivot_heat.index, color_continuous_scale='YlOrRd', text_auto=True, aspect="auto")
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, width='stretch')
    
    buf_heat = io.BytesIO()
    try:
        fig_heat.write_image(buf_heat, format='png')
        buf_heat.seek(0)
        st.download_button("📥 Скачать тепловую карту (PNG)", buf_heat, file_name="тепловая_карта_потерь.png", mime="image/png")
    except Exception as e:
        st.warning(f"⚠️ Экспорт PNG недоступен: {str(e)}")
    
    st.markdown("---")
    
    # УЛУЧШЕННАЯ ДЕТЕКЦИЯ АНОМАЛИЙ
    st.subheader("⚠️ Улучшенная детекция аномалий (Isolation Forest + признаки)")
    
    df_anom = df.copy()
    df_anom['ДеньНедели'] = df_anom['Дата'].dt.weekday
    df_anom['Месяц'] = df_anom['Дата'].dt.month
    df_anom['ЛогПотерь'] = np.log1p(df_anom['СуммаПотерь'])
    
    top_categories = df_anom['Категория'].value_counts().head(10).index
    top_stores = df_anom['Магазин'].value_counts().head(10).index
    df_anom['Категория_топ'] = df_anom['Категория'].where(df_anom['Категория'].isin(top_categories), 'Другие')
    df_anom['Магазин_топ'] = df_anom['Магазин'].where(df_anom['Магазин'].isin(top_stores), 'Другие')
    
    features = pd.get_dummies(df_anom[['ЛогПотерь', 'ДеньНедели', 'Месяц', 'Категория_топ', 'Магазин_топ']])
    
    if len(features) >= 10:
        model_anom = IsolationForest(contamination=0.05, random_state=42)
        df_anom['Аномалия'] = model_anom.fit_predict(features)
        аномалии = df_anom[df_anom['Аномалия'] == -1]
        
        if len(аномалии) > 0:
            аномалии_disp = аномалии.copy()
            аномалии_disp['Дата'] = аномалии_disp['Дата'].dt.strftime('%d.%m.%Y')
            st.dataframe(аномалии_disp[['Дата', 'Категория', 'СуммаПотерь', 'Магазин']], width='stretch')
            st.metric("Выявлено аномалий (улучшенный AI)", len(аномалии))
        else:
            st.success("✅ Улучшенный анализ: аномалий не выявлено")
    else:
        st.info("ℹ️ Недостаточно данных для улучшенной детекции аномалий")
    
    # Кластеризация
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
        st.subheader("🧩 Кластеры потерь (K-Means)")
        st.dataframe(кластеры, width='stretch')
    
    st.markdown("---")
    
    # ABC-АНАЛИЗ КАТЕГОРИЙ
    st.subheader("📊 ABC-анализ категорий по потерям")
    
    abc_data = суммарные_потери.copy()
    abc_data = abc_data.sort_values('СуммаПотерь', ascending=False)
    abc_data['Доля_%'] = abc_data['СуммаПотерь'] / abc_data['СуммаПотерь'].sum() * 100
    abc_data['Накопительная_доля'] = abc_data['Доля_%'].cumsum()
    
    def assign_abc(cum_pct):
        if cum_pct <= 80:
            return 'A'
        elif cum_pct <= 95:
            return 'B'
        else:
            return 'C'
    
    abc_data['ABC_класс'] = abc_data['Накопительная_доля'].apply(assign_abc)
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(abc_data[['Категория', 'СуммаПотерь', 'Доля_%', 'Накопительная_доля', 'ABC_класс']].round(2), width='stretch')
    with col2:
        fig_abc = px.bar(abc_data, x='Категория', y='Накопительная_доля',
                         color='ABC_класс', color_discrete_map={'A': '#ef4444', 'B': '#f59e0b', 'C': '#10b981'},
                         title='Накопительная доля потерь')
        fig_abc.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% (A)")
        fig_abc.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="95% (A+B)")
        st.plotly_chart(fig_abc, width='stretch')
    
    st.info("**A-класс** — 80% потерь, критичные категории. **B** — средний приоритет. **C** — низкий.")
    
    # ПРОГНОЗ ПО ТОП-3 КАТЕГОРИЯМ
    st.markdown("---")
    st.subheader("📈 Прогноз потерь по топ-3 категориям (Prophet)")
    
    top3_categories = суммарные_потери.head(3)['Категория'].tolist()
    ежедневные_потери = df.groupby('Дата')['СуммаПотерь'].sum().reset_index()
    
    if len(top3_categories) >= 1 and len(ежедневные_потери) >= 30:
        fig_multi = px.line(title='Прогноз потерь по топ-3 категориям на 7 дней')
        forecast_tables = {}
        
        for cat in top3_categories:
            df_cat = df[df['Категория'] == cat].groupby('Дата')['СуммаПотерь'].sum().reset_index()
            df_cat.columns = ['ds', 'y']
            
            if len(df_cat) >= 14:
                m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
                m.fit(df_cat)
                future_cat = m.make_future_dataframe(periods=7)
                forecast_cat = m.predict(future_cat)
                
                fig_multi.add_scatter(x=df_cat['ds'], y=df_cat['y'], mode='lines+markers', name=f'{cat} (факт)')
                fig_multi.add_scatter(x=forecast_cat['ds'], y=forecast_cat['yhat'], mode='lines', name=f'{cat} (прогноз)', line=dict(dash='dash'))
                fig_multi.add_scatter(x=forecast_cat['ds'], y=forecast_cat['yhat_lower'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False)
                fig_multi.add_scatter(x=forecast_cat['ds'], y=forecast_cat['yhat_upper'], fill='tonexty', mode='lines', fillcolor='rgba(255,99,71,0.2)', line_color='rgba(0,0,0,0)', showlegend=False)
                
                fc_table = forecast_cat[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).round(2)
                fc_table['ds'] = fc_table['ds'].dt.strftime('%d.%m.%Y')
                fc_table = fc_table.rename(columns={'ds': 'Дата', 'yhat': 'Прогноз', 'yhat_lower': 'Мин', 'yhat_upper': 'Макс'})
                forecast_tables[cat] = fc_table
        
        fig_multi.update_layout(height=600, yaxis_title='Сумма потерь, ₽')
        st.plotly_chart(fig_multi, width='stretch')
        
        if forecast_tables:
            for cat, table in forecast_tables.items():
                st.markdown(f"**Прогноз для категории: {cat}**")
                st.dataframe(table, width='stretch')
    else:
        st.info("ℹ️ Недостаточно данных для прогноза по топ-3 категориям")
    
    # Прогноз общей суммы (старый)
    ежедневные_потери.columns = ['ds', 'y']
    if len(ежедневные_потери) >= 14:
        st.subheader("📈 Общий прогноз на 7 дней (Prophet)")
        модель_prophet = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        модель_prophet.fit(ежедневные_потери)
        future = модель_prophet.make_future_dataframe(periods=7)
        forecast = модель_prophet.predict(future)
        
        fig_prog = px.line(ежедневные_потери, x='ds', y='y', title='Общий прогноз', labels={'y': 'Сумма потерь, ₽'})
        fig_prog.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз', line=dict(color='#f87171', dash='dash'))
        fig_prog.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Мин', line=dict(color='rgba(0,0,0,0)'), showlegend=False)
        fig_prog.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Макс', fill='tonexty', fillcolor='rgba(248, 113, 113, 0.2)', line=dict(color='rgba(0,0,0,0)'), showlegend=False)
        fig_prog.update_layout(height=600)
        st.plotly_chart(fig_prog, width='stretch')
        
        buf_prog = io.BytesIO()
        try:
            fig_prog.write_image(buf_prog, format='png')
            buf_prog.seek(0)
            st.download_button("📥 Скачать общий прогноз (PNG)", buf_prog, file_name="прогноз_потерь.png", mime="image/png")
        except Exception as e:
            st.warning(f"⚠️ Экспорт PNG недоступен: {str(e)}")
    
    # Персонализированные рекомендации (обновлённые с учётом ABC)
    top_category = суммарные_потери.iloc[0]['Категория'] if not суммарные_потери.empty else "нет данных"
    top_category_loss = суммарные_потери.iloc[0]['СуммаПотерь'] if not суммарные_потери.empty else 0
    top_store = потери_по_магазинам.iloc[0]['Магазин'] if not потери_по_магазинам.empty else "нет данных"
    top_store_loss = потери_по_магазинам.iloc[0]['СуммаПотерь'] if not потери_по_магазинам.empty else 0
    
    high_cluster_category = df[df['Кластер'] == 'Высокий']['Категория'].mode()[0] if 'Кластер' in df.columns and 'Высокий' in df['Кластер'].values else "нет данных"
    peak_day = pivot_heat.sum().idxmax() if 'pivot_heat' in locals() and not pivot_heat.empty else "нет данных"
    
    potential_save_min = round(текущие_потери * 0.20)
    potential_save_max = round(текущие_потери * 0.30)
    
    рекомендации = [
        f"🔴 **Высокий приоритет:** Усилить контроль в категории «{top_category}» (лидер потерь: {top_category_loss:,.0f} ₽). Ежедневные проверки приёмки и хранения.",
        f"🔴 **Высокий приоритет:** Аудит магазина «{top_store}» ({top_store_loss:,.0f} ₽ потерь). Проверить списание и безопасность.",
        f"🟡 **Средний приоритет:** Фокус на Высоком кластере (чаще всего «{high_cluster_category}»). Увеличить инвентаризации на 50%.",
        f"🟡 **Средний приоритет:** Пик потерь — «{peak_day}». Дополнительные проверки полок в этот день.",
        f"🟢 **Проактивно:** По ABC-анализу приоритет на A-классе — там 80% потерь.",
        f"💰 **Потенциальная экономия:** 20–30% от текущих потерь ({potential_save_min:,.0f} – {potential_save_max:,.0f} ₽)."
    ]
    
    st.subheader("💡 Персонализированные AI-рекомендации")
    for rec in рекомендации:
        st.markdown(rec)
    
    # Экспорт отчёта
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_export = df.copy()
        df_export['Дата'] = df_export['Дата'].dt.strftime('%d.%m.%Y')
        df_export.to_excel(writer, sheet_name='ИсходныеДанные', index=False)
        суммарные_потери.to_excel(writer, sheet_name='ПоКатегориям', index=False)
        потери_по_магазинам.to_excel(writer, sheet_name='ПоМагазинам', index=False)
        abc_data.to_excel(writer, sheet_name='ABC_анализ', index=False)
        if len(аномалии) > 0 if 'аномалии' in locals() else False:
            аномалии.to_excel(writer, sheet_name='Аномалии', index=False)
        if 'кластеры' in locals():
            кластеры.to_excel(writer, sheet_name='Кластеры', index=False)
        pd.DataFrame(рекомендации, columns=['Рекомендация']).to_excel(writer, sheet_name='Рекомендации', index=False)
    buffer.seek(0)
    
    st.download_button(
        label="📥 Скачать полный отчёт (Excel)",
        data=buffer,
        file_name=f"отчет_потерь_{datetime.today().strftime('%d.%m.%Y')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Основной поток
if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = xls.sheet_names[0]
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
        df = detect_columns(df)
        выполнить_анализ(df)
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {str(e)}")
elif st.session_state.get('use_test', False):
    df = сгенерировать_тестовые_данные()
    df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
    выполнить_анализ(df)
    st.session_state.use_test = False
else:
    st.info("👆 Загрузите файл Excel или нажмите «Тестовые данные» для демо.")
    st.markdown("### Превью тестовых данных")
    preview_df = сгенерировать_тестовые_данные()
    st.dataframe(preview_df.head(20), width='stretch')