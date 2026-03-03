chmod 751 /cluster/scratch/aoezkan
chmod 755 /cluster/scratch/aoezkan/tmp
setfacl -R -m u:ayavuz:rx /cluster/scratch/aoezkan/tmp
setfacl -d -m u:ayavuz:rx /cluster/scratch/aoezkan/tmp