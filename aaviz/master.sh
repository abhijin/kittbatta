#!/bin/bash
function choropleth(){ # spatial visualization of FAO communities
    python ../aaviz/choropleth.py \
        --nodes ../examples/world_communities.csv \
        --shape world \
        --node_column Node \
        --shape_column ISO_A3_EH \
        --community_column Community \
        --title "Example World Choropleth" \
        --o world_choropleth.pdf
}

function comm_spatial_faf(){   # spatial visualization of FAF communities
    python ../scripts/spatial_viz_community.py \
        --nodes ../work/C2C_faf_cereal_t100_scc1_cst0_xval0.5_nbins16_pp20_ncrux100-eb-ecut-labels.csv  \
        --shape ../../faf/data/cfs/2017_CFS_Metro_Areas_with_FAF.shp \
        --node_column Id \
        --shape_column FAF_Zone \
        --nodes_to_exclude 20 151 159 \
        --community_column Community \
        --title "Spatial viz FAF Cereal" \
        --o spatial_viz_faf_cereal_t100_rel.png
}

if [[ $# == 0 ]]; then
   echo "Here are the options:"
   grep "^function" $BASH_SOURCE | sed -e 's/function/  /' -e 's/[(){]//g' -e '/IGNORE/d'
else
   eval $1 $2
fi

