
OUT_DIR=deepbank-ace
mkdir -p $OUT_DIR
MEM=16000

for name in dev test; do
  ace -g $ERG_DIR/erg-1214-x86-64-0.9.24.dat deepbank-dmrs/$name.raw -1Tf --maxent=$ERG_DIR/wsj.mem --max-unpack-megabytes=$MEM --max-chart-megabytes=$MEM > ${OUT_DIR}/${name}.mrs 2> ${OUT_DIR}/${name}.log 

  # Extracts DMRS and evaluates.
  for TYPE in dmrs eds; do 
    MRS_WDIR=deepbank-${TYPE}-working
    python $HOME/DeepDeepParser/mrs/extract_ace_mrs.py $name $OUT_DIR -${TYPE}
    python $HOME/DeepDeepParser/mrs/eval_edm.py ${MRS_WDIR}/${name}.${TYPE}.edm ${OUT_DIR}/${name}.${TYPE}.edm
    python $HOME/smatch/smatch.py -f ${MRS_WDIR}/${name}.${TYPE}.amr ${OUT_DIR}/${name}.${TYPE}.amr
  done
done

