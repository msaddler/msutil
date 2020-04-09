user=$(whoami)

singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
-B /om2/user/$user/hearinglossnet/ibmHearingAid:/code_location \
/om2/user/$user/singularity-images/tfv1.13_probability.simg \
/om2/user/$user/jupyter_notebook_job.sh
