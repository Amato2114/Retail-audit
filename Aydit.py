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
    сегодня = datetime.now()  # Изменено на динамическую дату для актуальности
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
    
    # Приводим имена колонок к нижнему регистру для поиска
    lower_columns = {col.lower(): col for col in df.columns}
    
    # Кандидаты по ключевым словам
    date_candidates = [col for col in lower_columns if 'дат' in col or 'date' in col]
    category_candidates = [col for col in lower_columns if 'кат' in col or 'cat' in col or 'товар' in col or 'продукт' in col]
    loss_candidates = [col for col in lower_columns if 'пот' in col or 'loss' in col or 'сум' in col or 'убыт' in col or 'спис' in col]
    store_candidates = [col for col in lower_columns if 'маг' in col or 'store' in col or 'филиал' in col]
    
    # Если не нашли по словам — по типам данных
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
    
    # Выбираем первые кандидаты
    if not date_candidates or not category_candidates or not loss_candidates or not store_candidates:
        st.error("❌ Не удалось автоматически распознать все обязательные колонки. Проверьте файл или переименуйте колонки ближе к 'Дата', 'Категория', 'СуммаПотерь', 'Магазин'.")
        st.stop()
    
    date_col = lower_columns[date_candidates[0]]
    category_col = lower_columns[category_candidates[0]]
    loss_col = lower_columns[loss_candidates[0]]
    store_col = lower_columns[store_candidates[0]]
    
    st.success(f"✅ Автоматически распознаны колонки:\n- Дата: **{date_col}**\n- Категория: **{category_col}**\n- СуммаПотерь: **{loss_col}**\n- Магазин: **{store_col}**")
    
    # Переименовываем для унификации
    df = df.rename(columns={
        date_col: 'Дата',
        category_col: 'Категория',
        loss_col: 'СуммаПотерь',
        store_col: 'Магазин'
    })
    
    return df

# Функция анализа (полная)
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
    
    # Аномалии (scikit-learn)
    df['Индекс'] = np.arange(len(df))
    X_anom = df[['Индекс', 'СуммаПотерь']].values
    модель_anom = IsolationForest(contamination=0.1, random_state=42)
    df['Аномалия'] = модель_anom.fit_predict(X_anom)
    аномалии = df[df['Аномалия'] == -1]
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Общие потери", f"{текущие_потери:,.0f} ₽", delta=f"{изменение:+.1f}% vs прошлый")
    with col2:
        st.metric("Категорий", df['Категория'].nunique())
    with col3:
        st.metric("Магазинов", df['Магазин'].nunique())
    with col4:
        st.metric("Аномалий (AI)", len(аномалии))
    
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
    except ValueError as e:
        st.warning(f"⚠️ Не удалось экспортировать график: {str(e)}. Убедитесь, что Kaleido установлен.")
    
    # 🌡️ Тепловая карта БЕЗ locale
    st.subheader("🌡️ Тепловая карта потерь (категории × дни недели)")
    df_heat = df.copy()
    # map weekday -> русские названия, вместо day_name(locale='ru_RU')
    day_map = {
        0: 'Понедельник',
        1: 'Вторник',
        2: 'Среда',
        3: 'Четверг',
        4: 'Пятница',
        5: 'Суббота',
        6: 'Воскресенье',
    }
    df_heat['ДеньНедели'] = df_heat['Дата'].dt.weekday.map(day_map)
    pivot_heat = df_heat.pivot_table(
        values='СуммаПотерь',
        index='Категория',
        columns='ДеньНедели',
        aggfunc='sum',
        fill_value=0
    )
    дни_порядок = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    pivot_heat = pivot_heat.reindex(columns=дни_порядок)
    
    fig_heat = px.imshow(
        pivot_heat.values,
        x=дни_порядок,
        y=pivot_heat.index,
        color_continuous_scale='YlOrRd',
        text_auto=True,
        aspect="auto"
    )
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, width='stretch')
    
    buf_heat = io.BytesIO()
    try:
        fig_heat.write_image(buf_heat, format='png')
        buf_heat.seek(0)
        st.download_button("📥 Скачать тепловую карту (PNG)", buf_heat, file_name="тепловая_карта_потерь.png", mime="image/png")
    except ValueError as e:
        st.warning(f"⚠️ Не удалось экспортировать график: {str(e)}. Убедитесь, что Kaleido установлен.")
    
    # Аномалии
    if len(аномалии) > 0:
        st.subheader("⚠️ Выявленные аномалии (Isolation Forest)")
        аномалии_disp = аномалии.copy()
        аномалии_disp['Дата'] = аномалии_disp['Дата'].dt.strftime('%d.%m.%Y')
        st.dataframe(аномалии_disp[['Дата', 'Категория', 'СуммаПотерь', 'Магазин']], width='stretch')
    else:
        st.success("✅ Аномалий не выявлено")
    
    # Кластеризация (scikit-learn)
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
    
    # Прогноз с Prophet
    ежедневные_потери = df.groupby('Дата')['СуммаПотерь'].sum().reset_index()
    ежедневные_потери.columns = ['ds', 'y']
    
    прогноз_df = None
    if len(ежедневные_потери) >= 14:
        st.subheader("📈 Динамика и прогноз на 7 дней (Prophet)")
        модель_prophet = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        модель_prophet.fit(ежедневные_потери)
        future = модель_prophet.make_future_dataframe(periods=7)
        forecast = модель_prophet.predict(future)
        прогноз_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).rename(columns={'ds': 'Дата', 'yhat': 'Прогноз'}).round(2)
        
        fig_prog = px.line(ежедневные_потери, x='ds', y='y', title='Динамика потерь с AI-прогнозом (Prophet)', labels={'y': 'Сумма потерь, ₽'})
        fig_prog.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз', line=dict(color='#f87171', dash='dash'))
        fig_prog.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Мин', line=dict(color='rgba(0,0,0,0)'), showlegend=False)
        fig_prog.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Макс', fill='tonexty', fillcolor='rgba(248, 113, 113, 0.2)', line=dict(color='rgba(0,0,0,0)'), showlegend=False)
        fig_prog.update_layout(height=600)
        st.plotly_chart(fig_prog, width='stretch')
        
        buf_prog = io.BytesIO()
        try:
            fig_prog.write_image(buf_prog, format='png')
            buf_prog.seek(0)
            st.download_button("📥 Скачать график прогноза (PNG)", buf_prog, file_name="прогноз_потерь.png", mime="image/png")
        except ValueError as e:
            st.warning(f"⚠️ Не удалось экспортировать график: {str(e)}. Убедитесь, что Kaleido установлен.")
        
        прогноз_disp = прогноз_df.copy()
        прогноз_disp['Дата'] = прогноз_disp['Дата'].dt.strftime('%d.%m.%Y')
        прогноз_disp = прогноз_disp[['Дата', 'Прогноз', 'yhat_lower', 'yhat_upper']].rename(columns={'yhat_lower': 'Мин прогноз', 'yhat_upper': 'Макс прогноз'})
        st.dataframe(прогноз_disp, width='stretch')
    else:
        st.warning("⚠️ Недостаточно данных для прогноза (нужно ≥14 дней).")
    
    # Персонализированные рекомендации
    top_category = "нет данных"
    top_category_loss = 0
    if not суммарные_потери.empty:
        top_row = суммарные_потери.iloc[суммарные_потери['СуммаПотерь'].idxmax()]
        top_category = top_row['Категория']
        top_category_loss = top_row['СуммаПотерь']
    
    top_store = "нет данных"
    top_store_loss = 0
    if not потери_по_магазинам.empty:
        top_row_store = потери_по_магазинам.iloc[потери_по_магазинам['СуммаПотерь'].idxmax()]
        top_store = top_row_store['Магазин']
        top_store_loss = top_row_store['СуммаПотерь']
    
    high_cluster_category = "нет данных"
    if 'Кластер' in df.columns and 'Высокий' in df['Кластер'].values:
        high_cluster_mode = df[df['Кластер'] == 'Высокий']['Категория'].mode()
        high_cluster_category = high_cluster_mode[0] if len(high_cluster_mode) > 0 else "нет данных"
    
    peak_day = "нет данных"
    if 'pivot_heat' in locals() and not pivot_heat.empty:
        peak_day = pivot_heat.sum().idxmax()
    
    potential_save_min = round(текущие_потери * 0.20)
    potential_save_max = round(текущие_потери * 0.30)
    
    персонализированные_рекомендации = [
        f"🔴 **Высокий приоритет:** Усилить контроль в категории «{top_category}» — лидер по потерям ({top_category_loss:,.0f} ₽). Рекомендация: ежедневные проверки приёмки и хранения.",
        f"🔴 **Высокий приоритет:** Провести аудит магазина «{top_store}» — максимальные потери ({top_store_loss:,.0f} ₽). Проверить процедуры списания и безопасность.",
        f"🟡 **Средний приоритет:** Фокус на Высоком кластере (категория «{high_cluster_category}») — суммы выше среднего. Увеличить частоту инвентаризаций на 50%.",
        f"🟡 **Средний приоритет:** Оптимизировать по тепловой карте — пик потерь в «{peak_day}». Усилить контроль в этот день (дополнительные проверки полок).",
        f"🟢 **Проактивно:** Использовать прогноз для планирования — ожидаемый рост/спад на следующей неделе. Подготовить запас по категориям с риском.",
        f"💰 **Потенциальная экономия:** При внедрении рекомендаций — 20–30% от текущих потерь ({potential_save_min:,.0f} – {potential_save_max:,.0f} ₽ за период)."
    ]
    
    st.subheader("💡 Персонализированные AI-рекомендации по минимизации потерь")
    for rec in персонализированные_рекомендации:
        st.markdown(rec)
    
    # Полный отчёт
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_export = df.copy()
        df_export['Дата'] = df_export['Дата'].dt.strftime('%d.%m.%Y')
        df_export.to_excel(writer, sheet_name='ИсходныеДанные', index=False)
        суммарные_потери.to_excel(writer, sheet_name='ПоКатегориям')
        потери_по_магазинам.to_excel(writer, sheet_name='ПоМагазинам')
        if len(аномалии) > 0:
            аномалии_exp = аномалии.copy()
            аномалии_exp['Дата'] = аномалии_exp['Дата'].dt.strftime('%d.%m.%Y')
            аномалии_exp.to_excel(writer, sheet_name='Аномалии', index=False)
        if 'кластеры' in locals():
            кластеры.to_excel(writer, sheet_name='Кластеры')
        if прогноз_df is not None:
            прогноз_exp = прогноз_df.copy()
            прогноз_exp['Дата'] = прогноз_exp['Дата'].dt.strftime('%d.%m.%Y')
            прогноз_exp.to_excel(writer, sheet_name='Прогноз')
        pd.DataFrame(персонализированные_рекомендации, columns=['Рекомендация']).to_excel(writer, sheet_name='Рекомендации')
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
        # Чтение всех листов, берём первый с данными
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = xls.sheet_names[0]
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
        df = detect_columns(df)
        выполнить_анализ(df)
    except Exception as e:
        st.error(f"❌ Ошибка при чтении или распознавании файла: {str(e)}")
        st.info("💡 Проверьте формат Excel и наличие обязательных колонок")
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