# Progress and organisation for the Model-Satellite Intercomparison Project (MSGI)

## I. Background

**“How complete is cloud glaciation?” Nikolov, GRL, 2025**

In a previous study, I tracked **mixed-phase stratiform cloud tops** using CLAAS-3 data, which is based on the SEVIRI instrument onboard Meteosat Second Generation satellites. For each pixel, the data provides binary information on whether the cloud top is in the ice or liquid phase.

For each track, hereafter referred to as a **cloud**, I analysed the fraction of the cloud-top area in the ice phase at each timestep. I call this variable the **ice fraction**. It ranges from 0 to 1. A time series of ice fraction measurement gives information about the cloud top phase evolution.

Within each time series, we detect periods of increasing ice fraction. If the ice fraction increases by more than 0.4 during one of these periods, we classify the event as **glaciation**. When a glaciation event is detected, we identify the maximum ice fraction reached during that period, which we call the **peak ice fraction**.

**Our main result is that the peak ice fraction (IF) usually does not exceed 0.9: glaciation is incomplete. Instead, glaciation events typically peak at lower values, with a median peak IF of 0.7.** A cloud can glaciate multiple times, but this occurs rarely.

Tracking is performed at a resolution of approximately 15 km per pixel, every 15 minutes, for latitudes from 30° to 72° S/N and longitudes from -72° to 72° E.

### Tracking methodology

We selected six mixed-phase temperature bins: 0 to -6 °C, -6 to -12 °C, -12 to -18 °C, and so on down to -36 °C. From the CLAAS image at each timestep we get all the cloud tops in each temperature bin. Within each single-bin image, we then identify continuous objects and track them over time to obtain clouds. As stated in the paper:

> When given a time series (image of cloud tops in a single temperature bin), the tracker first identifies continuous regions of connected cloudy pixels as features. It then searches for spatial overlap between the features identified in consecutive timesteps. If a sufficient fraction of two such features overlap, they are marked as parts of the same track. Merging multiple features into a single track defines a “cloud” in the PE dataset. In this study, we therefore define a cloud as a time series of overlapping features, each representing a contiguous area of similar CTT.

The code for this processing is available on [GitHub](https://github.com/DragomirNikolov/Glaciation_time_estimator) as an installable Python library. More importantly, all tracked-cloud data are available as pandas DataFrames on [Zenodo](https://doi.org/10.5281/zenodo.17861542) and can be opened with `pd.read_parquet()`.

## II. Model-Satellite Intercomparison Project (MSGI)

MSGI is devoted to answering three questions that arise from **“How complete is cloud glaciation?”**

Because we needed cloud observations that are consistent in time and space, cloud tracking in **“How complete is cloud glaciation?”** is based on the CLAAS-3 dataset. This geostationary passive radiometer dataset has several disadvantages. First, the cloud mask and cloud-phase outputs are relatively low quality: they are noisy, unreliable, and likely biased towards the water phase. Second, they only describe cloud-top properties. We wanted to address these two limitations one by one.

### Is glaciation really incomplete?

#### Done

To answer this question, we colocate the tracked clouds in CLAAS-3 with measurements from two other datasets based on low-Earth-orbit (LEO) satellites: MODIS and DARDAR. LEO satellites irregularly observe narrow swaths of Earth, so we usually only have colocated observations from one timestep of a few cloud track. The colocations also do not cover the entire cloud, but only a line through it.

Nevertheless, we find that the IF measured by the different satellites is positively correlated. Peaks of IF in CLAAS-3 are also peaks of IF in MODIS and DARDAR. DARDAR predicts an overall higher ice fraction at every timestep. From its retrieval algorithm, we know that DARDAR is biased towards ice. MODIS and CLAAS, on the other hand, are biased towards liquid. We can therefore use the satellite measurements as reasonable bounds for what we want to confirm.

**Between 45% and 80% of clouds do not glaciate completely (IF > 0.9). Thus, the results from the first paper are confirmed.**

#### To-do

Optional: A team in Leipzig is tracking clouds again in the CLAAS-3 data, but they do not split the frame into individual temperature bins. Instead, they track clouds from the original CLAAS-3 image. I want to run my glaciation detection algorithms on their clouds and see whether my results remain the same.

### Why is glaciation incomplete?

After glaciation peaks, there is a drop in ice fraction. We suspect that this is due to ice crystal sedimentation. Using the DARDAR colocated measurements described above, we are now looking for increases in IWC 900 m below the cloud top at the timestep after peak IF. 900 m corresponds to 15 minutes multiplied by an assumed fall speed of 1 m/s.

#### Done

I have the code to extract the below-cloud-top data at the relevant locations, as well as the locations of the colocated pixels for one year.

#### To-do

Perform a statistical analysis to prove or disprove whether sedimentation or an increase in WBF below the cloud top explains the observed decrease in ice fraction.

### How do models simulate glaciation?

#### Done

Finally, we look at how glaciation is simulated within ICON-HAM. In ICON, we obtain snapshots of the cloud tops every 15 minutes by setting a TWC threshold of `1e-5`. For each pixel in the output, we get the cloud-top phase (`IWC/TWC` at the top cell) and cloud-top temperature (air temperature at the top cell). This data is then remapped to a grid equivalent to the CLAAS grid, and the exact same tracking procedure as for the CLAAS-3 dataset is applied (see [Background](#i-background)).

To understand the results of this analysis, we use several metrics:

1. **Peak ice fraction**: the maximum ice fraction reached during a glaciation event.
2. **Glaciation time**: in a time series, the time between a peak IF and the preceding IF minimum.
3. **Glaciation magnitude**: the difference between the peak IF and the preceding IF minimum.
4. **Glaciation occurrence rate**: the fraction of clouds that experience glaciation at all. Since we look for IF increases of 0.4 or more, we usually filter out clouds that never go below an IF of 0.6, because glaciation cannot be detected there.
5. **Number of glaciations per cloud**: as mentioned above, multiple glaciations per cloud are extremely rare in the observations (<1%). In the model, however, this is not the case: 10–15% of clouds glaciate more than once.

From the current analysis, we can conclude that in the main branch of ICON-HAM at R2B4, glaciation occurs:

1. **Too completely**: peak IF in the model is higher than observed in satellites.
2. **Too fast**: glaciation time is shorter than in satellite observations.
3. **Too often**: the occurrence rate in the model is much higher than in satellite observations.

#### To-do

What remains is to tune the relevant processes so that the model statistics better match the observations. All of the tasks below have been started, but we do not yet have conclusive results.

To do this, we should:

1. Test both versions of the WBF scheme: instant WBF (v1), rate-limited WBF (v3), and fully implicit WBF when available (v4). At R2B4:
    1. Rate limited WBF slows glaciation and brings glaciation time closer to observations
    2. Leads to a more complete glaciation (higher peak IF) - this is surprising and I still can't understand it.
2. Test these at different resolutions. The analysis is already done for one year at R2B4 and will be done by 01.06 for R2B5.
3. Test with different values of `fwbf` . Tests have been done at fwbf=1.0, 0.5, and 0. The lower it is, the better the results on all metrics. Even at `fwbf = 0`, too many clouds glaciate completely. 
4. Look at efects of model tuning on other variables. TOA Radiation and cloud cover.
5. Try to run the simulations nudged

## Notes

1. In general, I test on data from 2007. For the final paper, however, I want to use three years of data (2007, 2008, and 2009) for all analyses.
2. All simulation up until now have been ran without nudging. 
