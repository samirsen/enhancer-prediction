BT=${BT-../../bin/bedtools}
DATA=${DATA-../../data}

check()
{
	if diff $1 $2; then
    	echo ok
	else
    	echo fail
	fi
}

###########################################################
#  Test a basic self intersection. The relative distances
# should all be 0 in this case.
############################################################
echo "    reldist.t01...\c"
echo \
"reldist	count	total	fraction
0.00	43424	43424	1.000" > exp
$BT reldist -a $DATA/refseq.chr1.exons.bed.gz \
            -b $DATA/refseq.chr1.exons.bed.gz > obs
check obs exp
rm obs exp

###########################################################
#  Test intervaks that are randomly distributed. 
# The relative distances should equally represented .
############################################################
echo "    reldist.t02...\c"
echo \
"reldist	count	total	fraction
0.00	164	43408	0.004
0.01	551	43408	0.013
0.02	598	43408	0.014
0.03	637	43408	0.015
0.04	793	43408	0.018
0.05	688	43408	0.016
0.06	874	43408	0.020
0.07	765	43408	0.018
0.08	685	43408	0.016
0.09	929	43408	0.021
0.10	876	43408	0.020
0.11	959	43408	0.022
0.12	860	43408	0.020
0.13	851	43408	0.020
0.14	903	43408	0.021
0.15	893	43408	0.021
0.16	883	43408	0.020
0.17	828	43408	0.019
0.18	917	43408	0.021
0.19	875	43408	0.020
0.20	897	43408	0.021
0.21	986	43408	0.023
0.22	903	43408	0.021
0.23	944	43408	0.022
0.24	904	43408	0.021
0.25	867	43408	0.020
0.26	943	43408	0.022
0.27	933	43408	0.021
0.28	1132	43408	0.026
0.29	881	43408	0.020
0.30	851	43408	0.020
0.31	963	43408	0.022
0.32	950	43408	0.022
0.33	965	43408	0.022
0.34	907	43408	0.021
0.35	884	43408	0.020
0.36	965	43408	0.022
0.37	944	43408	0.022
0.38	911	43408	0.021
0.39	939	43408	0.022
0.40	921	43408	0.021
0.41	950	43408	0.022
0.42	935	43408	0.022
0.43	919	43408	0.021
0.44	915	43408	0.021
0.45	934	43408	0.022
0.46	843	43408	0.019
0.47	850	43408	0.020
0.48	1006	43408	0.023
0.49	937	43408	0.022" > exp
$BT reldist -a $DATA/refseq.chr1.exons.bed.gz \
            -b $DATA/aluY.chr1.bed.gz > obs
check obs exp
rm obs exp


###########################################################
#  Test intervaks that are consistently closer to one another
# than expected.  The distances should be biased towards 0.=
############################################################
echo "    reldist.t03...\c"
echo \
"reldist	count	total	fraction
0.00	20629	43422	0.475
0.01	2629	43422	0.061
0.02	1427	43422	0.033
0.03	985	43422	0.023
0.04	897	43422	0.021
0.05	756	43422	0.017
0.06	667	43422	0.015
0.07	557	43422	0.013
0.08	603	43422	0.014
0.09	487	43422	0.011
0.10	461	43422	0.011
0.11	423	43422	0.010
0.12	427	43422	0.010
0.13	435	43422	0.010
0.14	375	43422	0.009
0.15	367	43422	0.008
0.16	379	43422	0.009
0.17	371	43422	0.009
0.18	346	43422	0.008
0.19	389	43422	0.009
0.20	377	43422	0.009
0.21	411	43422	0.009
0.22	377	43422	0.009
0.23	352	43422	0.008
0.24	334	43422	0.008
0.25	315	43422	0.007
0.26	370	43422	0.009
0.27	330	43422	0.008
0.28	330	43422	0.008
0.29	280	43422	0.006
0.30	309	43422	0.007
0.31	326	43422	0.008
0.32	287	43422	0.007
0.33	294	43422	0.007
0.34	306	43422	0.007
0.35	307	43422	0.007
0.36	309	43422	0.007
0.37	271	43422	0.006
0.38	293	43422	0.007
0.39	311	43422	0.007
0.40	331	43422	0.008
0.41	320	43422	0.007
0.42	299	43422	0.007
0.43	327	43422	0.008
0.44	321	43422	0.007
0.45	326	43422	0.008
0.46	306	43422	0.007
0.47	354	43422	0.008
0.48	365	43422	0.008
0.49	336	43422	0.008
0.50	38	43422	0.001" > exp
$BT reldist -a $DATA/refseq.chr1.exons.bed.gz \
            -b $DATA/gerp.chr1.bed.gz > obs
check obs exp
rm obs exp
