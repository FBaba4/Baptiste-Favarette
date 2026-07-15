import os
import time
import json
import yfinance as yf
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────
# Liste des entreprises (tickers) — fortement élargie pour sortir d'un
# régime de petit échantillon (14 -> ~65 -> ici ~180). Plus le panel est
# large et diversifié par secteur/zone, plus les benchmarks sectoriels et
# les modèles prédictifs entraînés dessus ont un sens statistique.
# ─────────────────────────────────────────────────────────────────────────
tickers = [
    # --- Tech / Semi-conducteurs US ---
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "INTC", "AMD", "CSCO", "ORCL", "IBM", "ADBE", "CRM", "NFLX",
    "QCOM", "TXN", "AVGO", "MU", "ADI", "LRCX", "KLAC", "AMAT",
    "NOW", "INTU", "PANW", "SNOW", "PLTR", "UBER", "ABNB", "SHOP",

    # --- Finance US ---
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP",
    "SCHW", "BLK", "SPGI", "ICE", "CME", "PNC", "USB", "TFC",

    # --- Santé / Pharma US ---
    "JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "TMO", "ABT",
    "BMY", "AMGN", "GILD", "CVS", "MDT", "ISRG", "VRTX", "REGN",

    # --- Énergie US ---
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY",

    # --- Consommation US ---
    "WMT", "KO", "PEP", "PG", "MCD", "NKE", "HD", "DIS", "SBUX",
    "COST", "LOW", "TGT", "CL", "MDLZ", "MO", "PM", "EL",

    # --- Industrie / Auto / Aéronautique US ---
    "BA", "CAT", "GE", "F", "GM", "HON", "MMM", "LMT", "RTX",
    "DE", "UPS", "UNP", "NOC", "GD",

    # --- Télécom / Média US ---
    "T", "VZ", "TMUS", "CMCSA", "WBD",

    # --- Immobilier (REITs) US ---
    "PLD", "AMT", "EQIX", "SPG", "O",

    # --- CAC 40 et grandes valeurs françaises ---
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "BNP.PA", "AI.PA", "SU.PA",
    "DG.PA", "EL.PA", "RMS.PA", "KER.PA", "CAP.PA", "STLAP.PA", "BN.PA",
    "ENGI.PA", "VIE.PA", "ORA.PA", "ACA.PA", "GLE.PA", "GC.PA", "PUB.PA",
    "SGO.PA", "ML.PA", "RI.PA", "DSY.PA", "WLN.PA",

    # --- Allemagne ---
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BMW.DE",
    "MBG.DE", "VOW3.DE", "ADS.DE", "MRK.DE", "BAYN.DE", "DBK.DE",

    # --- Suisse ---
    "NESN.SW", "NOVN.SW", "ROG.SW", "UHR.SW", "ABBN.SW", "ZURN.SW",

    # --- Pays-Bas / Belgique ---
    "ASML.AS", "UNA.AS", "AD.AS", "INGA.AS", "ABI.BR",

    # --- Royaume-Uni ---
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "GSK.L", "BP.L", "DGE.L",

    # --- Espagne / Italie ---
    "SAN.MC", "ITX.MC", "IBE.MC", "ENEL.MI", "ISP.MI", "ENI.MI",

    # --- Scandinavie ---
    "NOVO-B.CO", "VOLV-B.ST", "ERIC-B.ST",
]

all_data = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # --- CROISSANCE DU CA (calcul manuel sur 2 exercices) ---
        croissance_ca_manuelle = "N/A"
        try:
            financials = stock.financials
            if "Total Revenue" in financials.index and financials.shape[1] >= 2:
                ca_recent = financials.loc["Total Revenue"].iloc[0]
                ca_precedent = financials.loc["Total Revenue"].iloc[1]
                if ca_recent and ca_precedent:
                    croissance_ca_manuelle = ((ca_recent - ca_precedent) / ca_precedent) * 100
        except Exception:
            pass

        # --- DATE DE CRÉATION ---
        start_date = info.get("startDate")
        if start_date:
            date_creation = str(pd.to_datetime(start_date, unit='s' if isinstance(start_date, int) else None).year)
        else:
            date_creation = "N/A"

        # --- VALORISATION ---
        pe_ratio = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        peg_ratio = info.get("trailingPegRatio", "N/A")
        pb_ratio = info.get("priceToBook", "N/A")
        debt_equity = info.get("debtToEquity", "N/A")
        ev_revenue = info.get("enterpriseToRevenue", "N/A")
        ev_ebitda = info.get("enterpriseToEbitda", "N/A")
        price_to_sales = info.get("priceToSalesTrailing12Months", "N/A")
        enterprise_value = info.get("enterpriseValue", "N/A")
        book_value = info.get("bookValue", "N/A")

        # --- MARGES ---
        marge_brute = info.get("grossMargins")
        marge_nette = info.get("profitMargins")
        marge_operationnelle = info.get("operatingMargins")
        marge_ebitda = info.get("ebitdaMargins")

        # --- RENDEMENT / RENTABILITÉ ---
        dividend_yield = info.get("dividendYield")
        yield_pct = f"{dividend_yield * 100:.2f}%" if dividend_yield else "N/A"

        payout_ratio = info.get("payoutRatio")
        payout_pct = f"{payout_ratio * 100:.1f}%" if payout_ratio else "N/A"

        roa = info.get("returnOnAssets")
        roa_pct = f"{roa * 100:.1f}%" if roa else "N/A"

        roe = info.get("returnOnEquity")
        roe_pct = f"{roe * 100:.1f}%" if roe else "N/A"

        # --- CROISSANCE (sources directes yfinance) ---
        revenue_growth = info.get("revenueGrowth")
        revenue_growth_pct = f"{revenue_growth * 100:.1f}%" if revenue_growth else "N/A"

        earnings_growth = info.get("earningsGrowth")
        earnings_growth_pct = f"{earnings_growth * 100:.1f}%" if earnings_growth else "N/A"

        earnings_q_growth = info.get("earningsQuarterlyGrowth")
        earnings_q_growth_pct = f"{earnings_q_growth * 100:.1f}%" if earnings_q_growth else "N/A"

        # --- LIQUIDITÉ / SOLIDITÉ FINANCIÈRE ---
        current_ratio = info.get("currentRatio", "N/A")
        quick_ratio = info.get("quickRatio", "N/A")
        total_cash = info.get("totalCash", "N/A")
        total_debt = info.get("totalDebt", "N/A")
        free_cashflow = info.get("freeCashflow", "N/A")
        operating_cashflow = info.get("operatingCashflow", "N/A")

        # --- RISQUE / MARCHÉ ---
        beta = info.get("beta", "N/A")
        eps_trailing = info.get("trailingEps", "N/A")
        eps_forward = info.get("forwardEps", "N/A")
        current_price = info.get("currentPrice", "N/A")
        target_mean_price = info.get("targetMeanPrice", "N/A")
        nb_analystes = info.get("numberOfAnalystOpinions", "N/A")
        recommandation = info.get("recommendationKey", "N/A")
        fifty_day_avg = info.get("fiftyDayAverage", "N/A")
        two_hundred_day_avg = info.get("twoHundredDayAverage", "N/A")
        fifty_two_week_high = info.get("fiftyTwoWeekHigh", "N/A")
        fifty_two_week_low = info.get("fiftyTwoWeekLow", "N/A")
        avg_volume_10j = info.get("averageVolume10days", "N/A")
        shares_outstanding = info.get("sharesOutstanding", "N/A")

        # --- ACTIONNARIAT ---
        insiders_pct = info.get("heldPercentInsiders")
        insiders_str = f"{insiders_pct * 100:.1f}%" if insiders_pct else "N/A"

        institutions_pct = info.get("heldPercentInstitutions")
        institutions_str = f"{institutions_pct * 100:.1f}%" if institutions_pct else "N/A"

        # --- IDENTITÉ MARCHÉ ---
        devise = info.get("currency", "N/A")
        bourse = info.get("fullExchangeName", info.get("exchange", "N/A"))

        data = {
            # --- Identité ---
            "Entreprise": info.get("longName", ticker),
            "Ticker": ticker,
            "Date création": date_creation,
            "Siège social": f"{info.get('address1', 'N/A')}, {info.get('city', 'N/A')}, {info.get('country', 'N/A')}",
            "Pays": info.get("country", "N/A"),
            "Ville": info.get("city", "N/A"),
            "Secteur": info.get("sector", "N/A"),
            "Industrie": info.get("industry", "N/A"),
            "Nombre employés": info.get("fullTimeEmployees", "N/A"),
            "Capitalisation (USD)": info.get("marketCap", "N/A"),
            "Chiffre d’affaires": info.get("totalRevenue", info.get("revenue", "N/A")),
            "Bénéfice Net": info.get("netIncomeToCommon", info.get("netIncome", "N/A")),
            "Devise": devise,
            "Bourse": bourse,

            # --- Marges & évolution ---
            "Marge brute (%)": f"{marge_brute * 100:.1f}%" if marge_brute else "N/A",
            "Marge nette (%)": f"{marge_nette * 100:.1f}%" if marge_nette else "N/A",
            "Croissance CA (%)": f"{croissance_ca_manuelle:.1f}%" if isinstance(croissance_ca_manuelle, (int, float)) else "N/A",
            "Marge opérationnelle (%)": f"{marge_operationnelle * 100:.1f}%" if marge_operationnelle else "N/A",
            "Marge EBITDA (%)": f"{marge_ebitda * 100:.1f}%" if marge_ebitda else "N/A",
            "Croissance CA yfinance (%)": revenue_growth_pct,
            "Croissance bénéfices (%)": earnings_growth_pct,
            "Croissance bénéfices trimestrielle (%)": earnings_q_growth_pct,

            # --- Valorisation ---
            "P/E Ratio (Valorisation)": f"{pe_ratio:.1f}" if isinstance(pe_ratio, (int, float)) else "N/A",
            "Forward P/E": f"{forward_pe:.1f}" if isinstance(forward_pe, (int, float)) else "N/A",
            "PEG Ratio": f"{peg_ratio:.2f}" if isinstance(peg_ratio, (int, float)) else "N/A",
            "P/B Ratio (Valo Bancaire)": f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else "N/A",
            "Valeur comptable/action": book_value,
            "Valeur d'entreprise (USD)": enterprise_value,
            "EV/Chiffre d'affaires": f"{ev_revenue:.2f}" if isinstance(ev_revenue, (int, float)) else "N/A",
            "EV/EBITDA": f"{ev_ebitda:.2f}" if isinstance(ev_ebitda, (int, float)) else "N/A",
            "Prix/Ventes (12 mois)": f"{price_to_sales:.2f}" if isinstance(price_to_sales, (int, float)) else "N/A",

            # --- Rendement / solidité ---
            "Rendement Dividende": yield_pct,
            "Ratio de distribution (Payout)": payout_pct,
            "Rentabilité Capitaux (ROE)": roe_pct,
            "Rentabilité Actifs (ROA)": roa_pct,
            "Ratio Dette/Capital (%)": f"{debt_equity:.1f}%" if isinstance(debt_equity, (int, float)) else "N/A",
            "Ratio liquidité générale": current_ratio,
            "Ratio liquidité immédiate": quick_ratio,
            "Trésorerie totale (USD)": total_cash,
            "Dette totale (USD)": total_debt,
            "Flux tréso. disponible (USD)": free_cashflow,
            "Flux tréso. opérationnel (USD)": operating_cashflow,

            # --- Risque & marché ---
            "Beta": beta,
            "BPA (trailing)": eps_trailing,
            "BPA (forward)": eps_forward,
            "Prix actuel": current_price,
            "Plus haut 52 sem.": fifty_two_week_high,
            "Plus bas 52 sem.": fifty_two_week_low,
            "Moyenne mobile 50j": fifty_day_avg,
            "Moyenne mobile 200j": two_hundred_day_avg,
            "Volume moyen 10j": avg_volume_10j,
            "Actions en circulation": shares_outstanding,
            "Objectif analystes (moyen)": target_mean_price,
            "Nombre d'analystes": nb_analystes,
            "Recommandation analystes": recommandation,

            # --- Actionnariat ---
            "Détention insiders (%)": insiders_str,
            "Détention institutionnels (%)": institutions_str,

            # --- Données brutes complètes ---
            # yfinance renvoie 80-100+ champs par ticker dans `info` ; on n'en
            # a sélectionné individuellement qu'une cinquantaine ci-dessus
            # (les plus pertinents pour l'analyse financière). Cette colonne
            # conserve TOUT le reste (et une redondance des champs déjà
            # extraits) sous forme de texte JSON, pour ne rien perdre — à
            # parser avec json.loads(...) si besoin d'un champ non repris
            # individuellement.
            "Données brutes (JSON)": json.dumps(info, default=str, ensure_ascii=False),
        }
        all_data.append(data)
        print(f"✅ {ticker} : OK")

    except Exception as e:
        print(f"❌ {ticker} : Erreur globale - {e}")

    time.sleep(0.5)  # pause légère anti-rate-limit Yahoo Finance (sans rapport avec le quota Gemini)

# Créer un DataFrame et sauvegarder en CSV
df = pd.DataFrame(all_data)

# Sécurité anti-doublon : au cas où un ticker apparaîtrait deux fois dans la
# liste ci-dessus par erreur de copier-coller.
avant = len(df)
df = df.drop_duplicates(subset=["Entreprise"], keep="first")
if len(df) < avant:
    print(f"\n⚠️ {avant - len(df)} doublon(s) supprimé(s) (même 'Entreprise' apparue plusieurs fois).")

df.to_csv("donnees_structurees.csv", index=False)

print(f"\n✅ {len(df)} entreprises collectées sur {len(tickers)} tickers demandés.")
print(f"📊 {len(df.columns)} colonnes/paramètres par entreprise.")
print("\n📋 Aperçu des données récoltées (sélection de colonnes) :")
columns_to_show = [
    "Entreprise",
    "Secteur",
    "Croissance CA (%)",
    "P/E Ratio (Valorisation)",
    "Rentabilité Capitaux (ROE)",
    "Rendement Dividende",
    "Marge brute (%)",
    "Marge nette (%)",
    "Croissance CA (%)",
    "Marge opérationnelle (%)",
    "Marge EBITDA (%)",
    "Croissance CA yfinance (%)",
    "Croissance bénéfices (%)",
    "Croissance bénéfices trimestrielle (%)",
    "Rendement Dividende",
    "Ratio de distribution (Payout)",
    "Rentabilité Capitaux (ROE)",
    "Rentabilité Actifs (ROA)",
    "Ratio Dette/Capital (%)",
    "Ratio liquidité générale",
    "Ratio liquidité immédiate",
    "Trésorerie totale (USD)",
    "Dette totale (USD)",
    "Flux tréso. disponible (USD)",
    "Flux tréso. opérationnel (USD)",
    "Valeur comptable/action",
    "Valeur d'entreprise (USD)",
    "EV/Chiffre d'affaires",
    "EV/EBITDA",
    "Prix/Ventes (12 mois)",
    "Beta",
    "BPA (trailing)",
    "BPA (forward)",
    "Prix actuel",
    "Plus haut 52 sem.",
    "Plus bas 52 sem.",
    "Moyenne mobile 50j",
    "Moyenne mobile 200j",
    "Volume moyen 10j",
    "Actions en circulation",
    "Objectif analystes (moyen)",
    "Nombre d'analystes",
    "Recommandation analystes",
    "Détention insiders (%)",
    "Détention institutionnels (%)",
]
print(df[columns_to_show].to_string(index=False))
print(df[columns_to_show].to_string(index=False))