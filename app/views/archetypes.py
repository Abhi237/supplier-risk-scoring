import plotly.express as px
import streamlit as st
from core import COLORWAY, DIVERGE, csv, scope_note, setup_page, summary, usd, vendors

s = summary()
cl = s["train_summary"]["clusters"]
setup_page("Indicative vendor archetypes", f"k-means on {len(s['config']['clustering']['features'])} standardised features · k = {cl['k']} "
           f"chosen by silhouette ({cl['silhouette']:.3f}) · {cl['n_clustered']:,} scored vendors with >= "
           f"{s['config']['clustering']['min_lines']} lines", section="vendor archetypes")
scope_note()
prof = csv("archetype_profiles.csv")
cols = st.columns(len(prof))
for c, (_, r) in zip(cols, prof.iterrows(), strict=True):
    c.metric(r.archetype, f"{int(r.n_vendors):,} vendors", f"{r.value_share:.0%} of value · score {r.risk_score:.0f}", delta_color="off")
sc = vendors().dropna(subset=["pc1"])
c1, c2 = st.columns([3, 2])
with c1:
    fig = px.scatter(sc, x="pc1", y="pc2", color="archetype", opacity=0.8, color_discrete_sequence=COLORWAY,
                     hover_data={"vendor": True, "n_lines": True, "late_rate_shrunk": ":.1%", "pc1": False, "pc2": False})
    fig.update_layout(height=440, title="Indicative archetypes in PCA space")
    st.plotly_chart(fig, width="stretch")
with c2:
    z = csv("archetype_centroids_z.csv").set_index("Unnamed: 0")
    fig = px.imshow(z, color_continuous_scale=DIVERGE, zmin=-2, zmax=2, text_auto=".1f", aspect="auto", labels=dict(color="z"))
    fig.update_layout(height=440, title="Centroid z-scores")
    st.plotly_chart(fig, width="stretch")
show = prof.copy()
show["value_usd"] = show.value_usd.map(usd)
show["value_at_risk"] = show.value_at_risk.map(usd)
st.dataframe(show.round(3), hide_index=True, width="stretch")
sel = csv("cluster_selection.csv")
fig = px.line(sel, x="k", y="silhouette", markers=True)
fig.update_layout(height=280, title="Silhouette by k (GMM BIC also reported in the table)")
c1, c2 = st.columns(2)
c1.plotly_chart(fig, width="stretch")
c2.dataframe(sel.round(3), hide_index=True, width="stretch")
st.info(f"These are **indicative archetypes** on only {cl['n_clustered']} vendors: three small clusters, one of which has "
        f"{int(prof.n_vendors.min())} vendors. Silhouette {cl['silhouette']:.3f} is moderate, but the labels name the dominant standardised "
        "feature and a different feature set or k would regroup vendors. Use them to frame a vendor conversation, not as hard categories.")
