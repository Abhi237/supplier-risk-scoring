import plotly.express as px
import streamlit as st
from core import COLORWAY, DIVERGE, brl, csv, proxy_note, sellers, setup_page, summary

s = summary()
cl = s["train_summary"]["clusters"]
setup_page("Indicative seller archetypes", f"k-means on 7 standardised reliability features · k = {cl['k']} chosen by silhouette "
           f"({cl['silhouette']:.3f}) · {cl['n_clustered']:,} sellers with >= {s['config']['clustering']['min_items']} items", section="seller archetypes")
proxy_note()
prof = csv("archetype_profiles.csv")
cols = st.columns(len(prof))
for c, (_, r) in zip(cols, prof.iterrows()):
    c.metric(r.archetype, f"{int(r.n_sellers):,} sellers", f"{r.gmv_share:.0%} of GMV · score {r.risk_score:.0f}",
             delta_color="off")
sc = sellers().dropna(subset=["pc1"])
c1, c2 = st.columns([3, 2])
with c1:
    fig = px.scatter(sc, x="pc1", y="pc2", color="archetype", opacity=0.7, color_discrete_sequence=COLORWAY,
                     hover_data={"seller_id": True, "n_items": True, "late_rate_shrunk": ":.1%", "pc1": False, "pc2": False})
    fig.update_layout(height=440, title="Indicative archetypes in PCA space")
    st.plotly_chart(fig, width="stretch")
with c2:
    z = csv("archetype_centroids_z.csv").set_index("Unnamed: 0")
    fig = px.imshow(z, color_continuous_scale=DIVERGE, zmin=-2, zmax=2, text_auto=".1f", aspect="auto",
                    labels=dict(color="z"))
    fig.update_layout(height=440, title="Centroid z-scores")
    st.plotly_chart(fig, width="stretch")
show = prof.copy()
show["gmv"] = show.gmv.map(brl)
show["gmv_at_risk"] = show.gmv_at_risk.map(brl)
st.dataframe(show.round(3), hide_index=True, width="stretch")
sel = csv("cluster_selection.csv")
fig = px.line(sel, x="k", y="silhouette", markers=True)
fig.update_layout(height=280, title="Silhouette by k (GMM BIC also reported in the table)")
c1, c2 = st.columns(2)
c1.plotly_chart(fig, width="stretch")
c2.dataframe(sel.round(3), hide_index=True, width="stretch")
st.info(f"These are **indicative archetypes**: silhouette {cl['silhouette']:.3f} means soft segments on a "
        "continuum, not well-separated groups. Use them to tailor the supplier conversation (dispatch SLA vs cancellation vs "
        "freight), not as hard categories.")
