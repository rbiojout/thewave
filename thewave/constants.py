#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path

DATABASE_DIR = path.realpath(__file__).\
    replace('constants.pyc','/marketdata/Data.db').\
    replace("constants.pyc","marketdata\\Data.db").\
    replace('constants.py','/marketdata/Data.db').\
    replace("constants.py","marketdata\\Data.db")
CONFIG_FILE_DIR = 'net_config.json'
LAMBDA = 1e-4  # lambda in loss function 5 in training
   # About time
NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
DAY = HOUR * 24
YEAR = DAY * 365
   # trading table name
TABLE_NAME = 'test'

# tickers SP500
SP500 = ('A','AAL','AAP','AAPL','ABBV','ABC','ABT','ACN','ADBE','ADI','ADM','ADP','ADS','ADSK','AEE','AEP','AES','AET','AFL','AGN','AIG','AIV','AIZ','AJG','AKAM','ALB','ALGN','ALK','ALL','ALLE','ALXN','AMAT','AMD','AME','AMG','AMGN','AMP','AMT','AMZN','ANDV','ANSS','ANTM','AON','AOS','APA','APC','APD','APH','ARE','ARNC','ATVI','AVB','AVGO','AVY','AWK','AXP','AYI','AZO','BA','BAC','BAX','BBT','BBY','BCR','BDX','BEN','BF.B','BHF','BHGE','BIIB','BK','BLK','BLL','BMY','BRK.B','BSX','BWA','BXP','C','CA','CAG','CAH','CAT','CB','CBG','CBOE','CBS','CCI','CCL','CDNS','CELG','CERN','CF','CFG','CHD','CHK','CHRW','CHTR','CI','CINF','CL','CLX','CMA','CMCSA','CME','CMG','CMI','CMS','CNC','CNP','COF','COG','COL','COO','COP','COST','COTY','CPB','CRM','CSCO','CSRA','CSX','CTAS','CTL','CTSH','CTXS','CVS','CVX','CXO','D','DAL','DE','DFS','DG','DGX','DHI','DHR','DIS','DISCA','DISCK','DISH','DLPH','DLR','DLTR','DOV','DPS','DRE','DRI','DTE','DUK','DVA','DVN','DWDP','DXC','EA','EBAY','ECL','ED','EFX','EIX','EL','EMN','EMR','EOG','EQIX','EQR','EQT','ES','ESRX','ESS','ETFC','ETN','ETR','EVHC','EW','EXC','EXPD','EXPE','EXR','F','FAST','FB','FBHS','FCX','FDX','FE','FFIV','FIS','FISV','FITB','FL','FLIR','FLR','FLS','FMC','FOX','FOXA','FRT','FTI','FTV','GD','GE','GGP','GILD','GIS','GLW','GM','GOOG','GOOGL','GPC','GPN','GPS','GRMN','GS','GT','GWW','HAL','HAS','HBAN','HBI','HCA','HCN','HCP','HD','HES','HIG','HLT','HOG','HOLX','HON','HP','HPE','HPQ','HRB','HRL','HRS','HSIC','HST','HSY','HUM','IBM','ICE','IDXX','IFF','ILMN','INCY','INFO','INTC','INTU','IP','IPG','IQV','IR','IRM','ISRG','IT','ITW','IVZ','JBHT','JCI','JEC','JNJ','JNPR','JPM','JWN','K','KEY','KHC','KIM','KLAC','KMB','KMI','KMX','KO','KORS','KR','KSS','KSU','L','LB','LEG','LEN','LH','LKQ','LLL','LLY','LMT','LNC','LNT','LOW','LRCX','LUK','LUV','LYB','M','MA','MAA','MAC','MAR','MAS','MAT','MCD','MCHP','MCK','MCO','MDLZ','MDT','MET','MGM','MHK','MKC','MLM','MMC','MMM','MNST','MO','MON','MOS','MPC','MRK','MRO','MS','MSFT','MSI','MTB','MTD','MU','MYL','NAVI','NBL','NCLH','NDAQ','NEE','NEM','NFLX','NFX','NI','NKE','NLSN','NOC','NOV','NRG','NSC','NTAP','NTRS','NUE','NVDA','NWL','NWS','NWSA','O','OKE','OMC','ORCL','ORLY','OXY','PAYX','PBCT','PCAR','PCG','PCLN','PDCO','PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PKG','PKI','PLD','PM','PNC','PNR','PNW','PPG','PPL','PRGO','PRU','PSA','PSX','PVH','PWR','PX','PXD','PYPL','QCOM','QRVO','RCL','RE','REG','REGN','RF','RHI','RHT','RJF','RL','RMD','ROK','ROP','ROST','RRC','RSG','RTN','SBAC','SBUX','SCG','SCHW','SEE','SHW','SIG','SJM','SLB','SLG','SNA','SNI','SNPS','SO','SPG','SPGI','SRCL','SRE','STI','STT','STX','STZ','SWK','SWKS','SYF','SYK','SYMC','SYY','T','TAP','TDG','TEL','TGT','TIF','TJX','TMK','TMO','TPR','TRIP','TROW','TRV','TSCO','TSN','TSS','TWX','TXN','TXT','UA','UAA','UAL','UDR','UHS','ULTA','UNH','UNM','UNP','UPS','URI','USB','UTX','V','VAR','VFC','VIAB','VLO','VMC','VNO','VRSK','VRSN','VRTX','VTR','VZ','WAT','WBA','WDC','WEC','WFC','WHR','WLTW','WM','WMB','WMT','WRK','WU','WY','WYN','WYNN','XEC','XEL','XL','XLNX','XOM','XRAY','XRX','XYL','YUM','ZBH','ZION','ZTS')