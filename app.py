# ==============================
# AI + إشارات + عرض نهائي
# ==============================

if not df.empty:

    # ==============================
    # تجهيز الداتا للموديل
    # ==============================
    df["target"] = (df["price"].shift(-3) > df["price"]).astype(int)
    df_ai = df.dropna()

    if len(df_ai) > 10:

        X = df_ai[["rsi","score"]]
        y = df_ai["target"]

        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        model.fit(X_train,y_train)

        acc = model.score(X_test,y_test)

        # ==============================
        # آخر بيانات لكل عملة
        # ==============================
        latest = df.sort_values("timestamp").groupby("coin").tail(1)
        X_latest = latest[["rsi","score"]]

        try:
            probs = model.predict_proba(X_latest)[:,1]
            latest["Chance %"] = probs * 100
        except:
            latest["Chance %"] = 0

        # ==============================
        # إشارات التداول (عربي)
        # ==============================
        def get_signal(score):
            if score >= 6:
                return "🚀 شراء قوي"
            elif score >= 4:
                return "🔥 شراء"
            elif score >= 2:
                return "⏳ انتظار"
            else:
                return "❌ رفض"

        latest["Signal"] = latest["score"].apply(get_signal)

        # ==============================
        # حالة البيانات (🟩🟨🟥)
        # ==============================
        counts = df.groupby("coin").size()

        def status_color(n):
            if n >= 20:
                return "🟩 كافي"
            elif n >= 10:
                return "🟨 متوسط"
            else:
                return "🟥 قليل"

        latest["Data Status"] = latest["coin"].apply(lambda c: status_color(counts.get(c, 0)))

        # ==============================
        # ترتيب حسب الأفضل
        # ==============================
        latest = latest.sort_values("Chance %", ascending=False)

        st.success(f"دقة الموديل: {round(acc*100,2)}%")

        # ==============================
        # تجهيز الجدول
        # ==============================
        display_df = latest[[
            "coin","price","drop_percent","rsi","volx","support","score","Signal","Chance %","Data Status"
        ]].rename(columns={
            "coin": "العملة",
            "price": "السعر",
            "drop_percent": "% الهبوط",
            "rsi": "RSI",
            "volx": "Vol X",
            "support": "الدعم",
            "score": "Score",
            "Signal": "الإشارة",
            "Chance %": "احتمال الصعود %",
            "Data Status": "حالة البيانات"
        })

        # ==============================
        # تلوين الإشارة
        # ==============================
        def color_signal(val):
            if "شراء قوي" in val:
                return "background-color: green; color: white"
            elif "شراء" in val:
                return "background-color: darkgreen; color: white"
            elif "انتظار" in val:
                return "background-color: orange"
            else:
                return "background-color: red; color: white"

        # ==============================
        # تلوين حالة البيانات
        # ==============================
        def color_data_status(val):
            if "🟩" in val:
                return "background-color: green; color: white"
            elif "🟨" in val:
                return "background-color: orange"
            else:
                return "background-color: red; color: white"

        # ==============================
        # عرض الجدول النهائي
        # ==============================
        st.dataframe(
            display_df.style
            .applymap(color_signal, subset=["الإشارة"])
            .applymap(color_data_status, subset=["حالة البيانات"]),
            use_container_width=True
        )

    else:
        st.warning("⚠️ البيانات غير كافية لتشغيل AI")
