import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import gridspec

try:
    from cmcrameri import cm
except ImportError:
    cm = None


DATA_PATH = "/Volumes/University_stuff/Code_Projects/MSGI/plotting/Bulk/bulk_data_filtered.parquet"
OUT_PATH = "/Volumes/University_stuff/Code_Projects/MSGI/plotting/Publication_figures/Bulk_bias_sources_figure.pdf"

l2_start = 5
l_depth = 7
ind_range = [-3, 4]
confidence_level = 0.90
n_bootstrap = 2000
bootstrap_seed = 42

inds = np.arange(ind_range[0], ind_range[1])
rel_times_min = np.array(inds * 15, dtype=int)

level_groups = [
    list(range(0, l2_start)),
    list(range(l2_start, l2_start + l_depth)),
    list(range(l2_start + l_depth, l2_start + 2 * l_depth)),
]

level_labels = [
    "Cloud top",
    f"-{l2_start * 120} to -{(l2_start + l_depth) * 120} m",
    f"-{(l2_start + l_depth) * 120} to -{(l2_start + 2 * l_depth) * 120} m",
]

cloud_type_labels = [
    "Altostratus",
    "Nimbostratus",
    "Stratocumulus",
    "Stratus",
]


def finite_layer_mask(profiles, group):
    """One boolean per cloud: True when this grouped vertical layer has data."""
    profile_matrix = np.vstack(profiles.to_numpy()).astype(float)
    return np.any(np.isfinite(profile_matrix[:, group]), axis=1)


def bootstrap_ci(values, statistic=np.nanmean, confidence=0.90, n_boot=2000, rng=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, np.nan

    point = statistic(values)

    if values.size == 1:
        return point, point, point

    if rng is None:
        rng = np.random.default_rng()

    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    boot_stats = statistic(samples, axis=1)

    alpha = 1.0 - confidence
    lo, hi = np.nanpercentile(
        boot_stats,
        [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
    )

    return point, lo, hi


def bootstrap_count_ci(mask, confidence=0.90, n_boot=2000, rng=None):
    mask = np.asarray(mask, dtype=bool)

    if mask.size == 0:
        return np.nan, np.nan, np.nan

    point = mask.sum()

    if mask.size == 1:
        return point, point, point

    if rng is None:
        rng = np.random.default_rng()

    samples = rng.choice(mask.astype(int), size=(n_boot, mask.size), replace=True)
    boot_counts = samples.sum(axis=1)

    alpha = 1.0 - confidence
    lo, hi = np.nanpercentile(
        boot_counts,
        [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
    )

    return point, lo, hi


def plot_heatmap_with_ci(
    ax,
    data,
    ci_low,
    ci_high,
    title,
    cbar_label,
    cmap,
    norm=None,
    value_format="{:.1f}",
    ci_format="{:.1f}",
):
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        origin="upper",
    )

    ax.set_title(title)
    ax.set_xticks(np.arange(len(rel_times_min)))
    ax.set_xticklabels(rel_times_min)
    ax.set_yticks(np.arange(len(level_labels)))
    ax.set_yticklabels(level_labels)
    ax.set_ylabel("Vertical level")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            lo = ci_low[i, j]
            hi = ci_high[i, j]

            if not np.isfinite(value):
                continue

            ax.text(
                j,
                i,
                f"{value_format.format(value)}\n[{ci_format.format(lo)}, {ci_format.format(hi)}]",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    return im


def plot_stacked_count_columns(
    ax,
    data,
    x,
    labels,
    title,
    legend_ax=None,
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bottoms = np.zeros(data.shape[1], dtype=float)
    bar_width = 10.0

    for i, label in enumerate(labels):
        values = np.nan_to_num(data[i], nan=0.0)
        bars = ax.bar(
            x,
            values,
            bottom=bottoms,
            width=bar_width,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )

        for bar, value, bottom in zip(bars, values, bottoms):
            if value <= 0:
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bottom + value / 2.0,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

        bottoms += values

    for xpos, total in zip(x, bottoms):
        ax.text(
            xpos,
            total,
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
        )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xlim(x.min() - 7.5, x.max() + 7.5)
    ax.set_ylabel("Clouds")
    ax.grid(axis="y", alpha=0.3)

    if legend_ax is None:
        ax.legend(fontsize=7, frameon=False)
    else:
        legend_ax.axis("off")
        legend_ax.legend(
            *ax.get_legend_handles_labels(),
            loc="center left",
            fontsize=7,
            frameon=False,
        )


def plot_lines_with_ci(
    ax,
    x,
    means,
    ci_low,
    ci_high,
    title,
    ylabel,
    ddc_label,
    avg_label,
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.plot(
        x,
        means["ddc"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        color=colors[0],
        linestyle="-",
        label=ddc_label,
    )
    ax.fill_between(
        x,
        ci_low["ddc"],
        ci_high["ddc"],
        color=colors[0],
        alpha=0.18,
        linewidth=0,
    )

    ax.plot(
        x,
        means["avg"],
        marker="s",
        markersize=3.5,
        linewidth=1.6,
        color=colors[1],
        linestyle="--",
        label=avg_label,
    )
    ax.fill_between(
        x,
        ci_low["avg"],
        ci_high["avg"],
        color=colors[1],
        alpha=0.14,
        linewidth=0,
    )

    ax.axvline(0, color="k", linewidth=1, alpha=0.35)
    ax.set_title(title)
    ax.set_xticks(rel_times_min)
    ax.set_xlim(rel_times_min.min() - 7.5, rel_times_min.max() + 7.5)
    
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

bulk_data_filtered = pd.read_parquet(DATA_PATH)

required_columns = {
    "rel_ind",
    "Z_bulk_means",
    "ddc_cloud_top_temp_means_K",
    "ddc_cloud_top_height_means_m",
    "avg_ctt",
    "avg_cth",
    "avg_cot",
    "3km_cot",
    "Cloud type",
}
missing_columns = required_columns.difference(bulk_data_filtered.columns)

if missing_columns:
    raise KeyError(f"Missing required columns in {DATA_PATH}: {sorted(missing_columns)}")

rng = np.random.default_rng(bootstrap_seed)
n_levels = len(level_groups)
n_times = len(inds)

n_map = np.full((n_levels, n_times), np.nan)
n_ci_low = np.full_like(n_map, np.nan)
n_ci_high = np.full_like(n_map, np.nan)

cloud_type_count_map = np.full((len(cloud_type_labels), n_times), np.nan)

ctt_ddc = np.full(n_times, np.nan)
ctt_ddc_ci_low = np.full(n_times, np.nan)
ctt_ddc_ci_high = np.full(n_times, np.nan)

ctt_avg = np.full(n_times, np.nan)
ctt_avg_ci_low = np.full(n_times, np.nan)
ctt_avg_ci_high = np.full(n_times, np.nan)

cth_ddc = np.full(n_times, np.nan)
cth_ddc_ci_low = np.full(n_times, np.nan)
cth_ddc_ci_high = np.full(n_times, np.nan)

cth_avg = np.full(n_times, np.nan)
cth_avg_ci_low = np.full(n_times, np.nan)
cth_avg_ci_high = np.full(n_times, np.nan)

cot_3km = np.full(n_times, np.nan)
cot_3km_ci_low = np.full(n_times, np.nan)
cot_3km_ci_high = np.full(n_times, np.nan)

cot_avg = np.full(n_times, np.nan)
cot_avg_ci_low = np.full(n_times, np.nan)
cot_avg_ci_high = np.full(n_times, np.nan)

for j, ind in enumerate(inds):
    time_df = bulk_data_filtered[bulk_data_filtered["rel_ind"] == ind]

    if time_df.empty:
        continue

    cloud_type_counts = time_df["Cloud type"].value_counts()
    for i, cloud_type in enumerate(cloud_type_labels):
        cloud_type_count_map[i, j] = cloud_type_counts.get(cloud_type, 0)

    ctt_c = time_df["ddc_cloud_top_temp_means_K"].astype(float).to_numpy() - 273.15
    avg_ctt_c = time_df["avg_ctt"].astype(float).to_numpy() - 273.15
    cth_m = time_df["ddc_cloud_top_height_means_m"].astype(float).to_numpy()
    avg_cth_m = time_df["avg_cth"].astype(float).to_numpy() /1000
    cot_3km_values = time_df["3km_cot"].astype(float).to_numpy()
    avg_cot_values = time_df["avg_cot"].astype(float).to_numpy()

    ctt_ddc[j], ctt_ddc_ci_low[j], ctt_ddc_ci_high[j] = bootstrap_ci(
        ctt_c,
        confidence=confidence_level,
        n_boot=n_bootstrap,
        rng=rng,
    )
    ctt_avg[j], ctt_avg_ci_low[j], ctt_avg_ci_high[j] = bootstrap_ci(
        avg_ctt_c,
        confidence=confidence_level,
        n_boot=n_bootstrap,
        rng=rng,
    )
    cth_ddc[j], cth_ddc_ci_low[j], cth_ddc_ci_high[j] = bootstrap_ci(
        cth_m,
        confidence=confidence_level,
        n_boot=n_bootstrap,
        rng=rng,
    )
    cth_avg[j], cth_avg_ci_low[j], cth_avg_ci_high[j] = bootstrap_ci(
        avg_cth_m,
        confidence=confidence_level,
        n_boot=n_bootstrap,
        rng=rng,
    )
    cot_3km[j], cot_3km_ci_low[j], cot_3km_ci_high[j] = bootstrap_ci(
        cot_3km_values,
        confidence=confidence_level,
        n_boot=n_bootstrap,
        rng=rng,
    )
    cot_avg[j], cot_avg_ci_low[j], cot_avg_ci_high[j] = bootstrap_ci(
        avg_cot_values,
        confidence=confidence_level,
        n_boot=n_bootstrap,
        rng=rng,
    )

    for i, group in enumerate(level_groups):
        valid_vertical = finite_layer_mask(time_df["Z_bulk_means"], group)

        n, n_lo, n_hi = bootstrap_count_ci(
            valid_vertical,
            confidence=confidence_level,
            n_boot=n_bootstrap,
            rng=rng,
        )

        n_map[i, j], n_ci_low[i, j], n_ci_high[i, j] = n, n_lo, n_hi

ci_percent = 100 * confidence_level
sample_cmap = "viridis" if cm is None else cm.batlow

fig = plt.figure(figsize=(10, 12), constrained_layout=True)
gs = gridspec.GridSpec(
    5,
    2,
    height_ratios=[1, 1, 1, 1, 1],
    width_ratios=[1, 0.05],
    figure=fig,
)

ax_n = fig.add_subplot(gs[0, 0])
ax_ctt = fig.add_subplot(gs[1, 0])
ax_cth = fig.add_subplot(gs[2, 0])
ax_cot_3km = fig.add_subplot(gs[3, 0])
ax_cloud_type = fig.add_subplot(gs[4, 0])
ax_n_cb = fig.add_subplot(gs[0, 1])
ax_cloud_type_legend = fig.add_subplot(gs[4, 1])

im_n = plot_heatmap_with_ci(
    ax_n,
    n_map,
    n_ci_low,
    n_ci_high,
    title=f"Valid samples per vertical layer ({ci_percent:g}% bootstrap CI)",
    cbar_label="Samples",
    cmap=sample_cmap,
    norm=Normalize(vmin=0, vmax=np.nanmax(n_map)),
    value_format="{:.0f}",
    ci_format="{:.0f}",
)

plot_lines_with_ci(
    ax_ctt,
    rel_times_min,
    means={"ddc": ctt_ddc, "avg": ctt_avg},
    ci_low={"ddc": ctt_ddc_ci_low, "avg": ctt_avg_ci_low},
    ci_high={"ddc": ctt_ddc_ci_high, "avg": ctt_avg_ci_high},
    title=f"Cloud-top temperature of contributing clouds ({ci_percent:g}% bootstrap CI)",
    ylabel="Cloud-top temperature [deg C]",
    ddc_label="DARDAR",
    avg_label="SEVIRI avg",
)

plot_lines_with_ci(
    ax_cth,
    rel_times_min,
    means={"ddc": cth_ddc, "avg": cth_avg},
    ci_low={"ddc": cth_ddc_ci_low, "avg": cth_avg_ci_low},
    ci_high={"ddc": cth_ddc_ci_high, "avg": cth_avg_ci_high},
    title=f"Cloud-top height of contributing clouds ({ci_percent:g}% bootstrap CI)",
    ylabel="Cloud-top height [km]",
    ddc_label="DARDAR",
    avg_label="SEVIRI avg",
)

plot_lines_with_ci(
    ax_cot_3km,
    rel_times_min,
    means={"ddc": cot_3km, "avg": cot_avg},
    ci_low={"ddc": cot_3km_ci_low, "avg": cot_avg_ci_low},
    ci_high={"ddc": cot_3km_ci_high, "avg": cot_avg_ci_high},
    title=f"Cloud optical thickness by time ({ci_percent:g}% bootstrap CI)",
    ylabel="Cloud optical thickness",
    ddc_label="3 km COT",
    avg_label="SEVIRI avg",
)

plot_stacked_count_columns(
    ax_cloud_type,
    cloud_type_count_map,
    rel_times_min,
    cloud_type_labels,
    title="Cloud-type counts by time",
    legend_ax=ax_cloud_type_legend,
)

for ax in (ax_n, ax_ctt, ax_cth, ax_cot_3km):
    ax.set_xlabel("")
ax_ctt.set_ylim(-36,0)
ax_cth.set_ylim(0,6)
ax_cloud_type.set_xlabel("Time relative to glaciation peak [min]")
ax_ctt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=7,
    frameon=False,
)
ax_cth.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=7,
    frameon=False,
)
ax_cot_3km.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=7,
    frameon=False,
)

fig.colorbar(im_n, cax=ax_n_cb).set_label("Samples")

fig.savefig(OUT_PATH, format="pdf", dpi=300)
plt.show()
