from netCDF4 import Dataset
fh=Dataset("scatteringTablesGPM_SPH.nc")
ng=272
zKuG=fh["zKuG"][0:272]
zKaG=fh["zKaG"][0:272]
attKaG=fh["attKaG"][0:272]
attKuG=fh["attKuG"][0:272]
dmG=fh["dmg"][:272]
gwc=fh["gwc"][:272]
graupRate=fh["graupRate"][:272]
zKuR=fh["zKuR"][0:289]
zKaR=fh["zKaR"][0:289]
attKaR=fh["attKaR"][0:289]
attKuR=fh["attKuR"][0:289]
dmR=fh["dmr"][:289]
rwc=fh["rwc"][:289]
rainRate=fh["rainRate"][:289]
ns=253
zKuS=fh["zKuS"][0:ns]
zKaS=fh["zKaS"][0:ns]
attKaS=fh["attKaS"][0:ns]
attKuS=fh["attKuS"][0:ns]
dmS=fh["dms"][:ns]
swc=fh["swc"][:ns]
snowRate=fh["snowRate"][:ns]

#['13.8','35','94','183.31','325.15','660.00']
fhssRG=Dataset("ssRG-scatteringTables.nc")
#iwcST=fhssRG["iwc"][:]
iwcST=fhssRG["iwc"][:]
dmST=fhssRG["dm"][:]
kextST=fhssRG["kext"][:]
kscaST=fhssRG["ksca"][:]
gST=fhssRG["g"][:]
zST=fhssRG["zT"][:]
