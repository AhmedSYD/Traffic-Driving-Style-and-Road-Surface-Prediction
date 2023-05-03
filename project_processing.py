import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import csv
import sys
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2_contingency
import os
import glob
import math
import seaborn as sn

pd.options.mode.chained_assignment = None

def chisq_of_df_cols(df, c1, c2):

    crosstab=pd.crosstab(df[c1], df[c2])
    # print("crose tab:",crosstab)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return(chi2_contingency(crosstab))

def get_nan_indexes(dataframe, attr_name):
    return dataframe[dataframe[attr_name].isnull()].index.tolist()

def draw_specific_attribute(dataframe,attribute_name,highlight_indexes=[],filename=None, filling_method="", visualize=False):
    price_attr = dataframe[attribute_name]

    plt.figure()
    price_attr.plot(x=list(range(1, 207)),marker='o', color='black')
    if(len(highlight_indexes)>0):
        # specific_val_dataframe=dataframe.loc[highlight_indexes,:]
        # df = pd.DataFrame({"indexes":highlight_indexes,"price":list(specific_val_dataframe['price'])})
        # df.plot.scatter(x="indexes",y="price")
        price_attr.loc[highlight_indexes].plot(x=list(range(1, 207)), marker='o', color='red',linestyle="None")
    if(filling_method!=""):
        plt.title(attribute_name+"_null_"+filling_method)
    else:
        plt.title(attribute_name)

    if(not(filename is None)):
        foldername=filename.split("/")[-2]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        plt.savefig(filename+ ".jpg")


    if(visualize):
        plt.show()

    plt.cla()
    plt.close()

def mean_by_group(dataframe, attr_name, indexes_list, number_of_vals=10,fill_dataFrame=False):
    n=len(list(dataframe[attr_name]))
    means=[]

    for index in indexes_list:
        if((index-(number_of_vals/2))<0):
            min_index =0
        else:
            min_index=index-(number_of_vals/2)
        # print("min_index=",min_index)
        if((index+(number_of_vals/2))>(n-1)):
            max_index = n-1
        else:
            max_index = index +(number_of_vals / 2)

        # print("max_index=", max_index)
        # print("group of values=",dataframe.loc[min_index:max_index,attr_name].dropna())
        temp=dataframe.loc[min_index:max_index,attr_name].dropna().mean()
        # print("temp=",temp)
        if(math.isnan(temp)):
            val=0
        else:
            val=round(temp)

        if fill_dataFrame:
            dataframe.loc[index,attr_name]=val
        # print("val=",val)
        means.append(val)
    if fill_dataFrame:
        return means,dataframe
    else:
        return means

def nan_analysis(dataframe, attr_name ,analysis_type="replace",value_name='', null_indexs=[],num_vals_mean=10):
    df_new=dataframe.copy()
    if (analysis_type=="replace"):
        if value_name=="mean":
            val = dataframe[attr_name].dropna().mean()
            # print("mean val=",val)
        elif value_name=="median":
            val = dataframe[attr_name].dropna().median()
            # print("median val=", val)
        elif value_name.startswith("mean_group"):
            ##get mean and fill it inside the fuction to make the runtime more faster
            num_vals = num_vals_mean
            val_list,df_new = mean_by_group(df_new, attr_name, null_indexs, number_of_vals=num_vals,fill_dataFrame = True)
            # print("mean_group val=", val_list)
        else:
            val=0
            print("no specified val=", val)
        ##fill data
        if(not(value_name.startswith("mean_group"))):
            df_new[attr_name]=df_new[attr_name].fillna(val)

    if(analysis_type=="drop"):
        df_new[attr_name]=df_new[attr_name].dropna()
    return df_new

def get_df_from_csv_files(csv_files_ls):
    print(csv_files_ls)
    all_data = []
    for path_file in csv_files_ls:
        with open(path_file, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=';', quotechar='|')
            # print("data", data)
            # print("type of data:", type(data))
            data_content = []
            for row in data:
                data_content.append(row)

        data_headers = data_content[0]
        del data_content[0]

        all_data += data_content

    df = pd.DataFrame(all_data, columns=data_headers)
    df_upd=df.replace(",",".",regex=True)
    df_upd=df_upd.replace("",np.nan)
    # condition= df_upd.columns != "roadSurface" or df_upd.columns != "traffic" or df_upd.columns != "drivingStyle"
    df_upd[df_upd.columns.difference(["roadSurface","traffic","drivingStyle"])] = \
        df_upd[df_upd.columns.difference(["roadSurface","traffic","drivingStyle"])].astype("float")
    # print("dtypes of df_upd:",df_upd.dtypes)
    # print("df[AltitudeVariation]=", df_upd["AltitudeVariation"])
    # print("df[EngineCoolantTemperature]=", df_upd["EngineCoolantTemperature"])

    # print("df_columns=", df.columns)
    return df_upd


def try_filling_null_methods(df, attributeName,visual=False):
    null_indexes = get_nan_indexes(df, attributeName)
    if(len(null_indexes)==0):
        print("no nulls in "+attributeName)
        return
    instructions = ["drop", "mean", "median", "mean_group_100","mean_group_200","mean_group_150"]
    for instruction in instructions:
        if instruction == "drop":
            df_new = nan_analysis(df, attributeName, analysis_type=instruction, value_name='')
        elif instruction == "mean_group":

            df_new = nan_analysis(df, attributeName, analysis_type="replace", value_name=instruction,\
                                  null_indexs=null_indexes, num_vals_mean=int(instruction.split("_")[-1]))
        else:

            df_new = nan_analysis(df, attributeName, analysis_type="replace", value_name=instruction,
                                  null_indexs=null_indexes)
        draw_specific_attribute(df_new, attributeName, highlight_indexes=null_indexes,\
                                filename="out_figures/" +attributeName+ "_nan_"+instruction,\
                                filling_method=instruction,visualize=visual)

def apply_fill_missing_method(df, attributeName,instruction="drop",num_vals_mean=10):
    null_indexes = get_nan_indexes(df, attributeName)
    if(len(null_indexes)==0):
        print("no nulls in "+attributeName)
        return df
    if not(instruction in ["drop", "mean", "median", "mean_group"]):
        print("instruction is not identified. Please, write it correctly")
        df_new=df
    elif instruction == "drop":
        df_new = nan_analysis(df, attributeName, analysis_type=instruction, value_name='')
    elif instruction == "mean_group":

        df_new = nan_analysis(df, attributeName, analysis_type="replace", value_name=instruction,
                              null_indexs=null_indexes,num_vals_mean=num_vals_mean)
    else:

        df_new = nan_analysis(df, attributeName, analysis_type="replace", value_name=instruction,
                              null_indexs=null_indexes)
    # draw_specific_attribute(df_new, attributeName, highlight_indexes=null_indexes, \
    #                         filename="out_figures/" + attributeName + "_nan_" + instruction, \
    #                         filling_method=instruction, visualize=True)
    return df_new

def cap_data(df):
    for col in df.columns:
        print("capping the ",col)
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
            percentiles = df[col].quantile([0.01,0.99]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col]=df[col]
    return df

def draw_box_plots(cols,df, foldername=None, filename=None, visualize=False):
    print("columns inside fn",df.columns)
    for i in range(0, len(cols), 1):
        fig = plt.figure()
        boxplt = df.boxplot(column=cols[i])
        # plt.show(boxplt)

        if(filename is None):
            plt.title(cols[i])
            if(foldername is None):
                fig.savefig(cols[i] + "_boxplot.jpg")
            else:
                fig.savefig(os.path.join(foldername,cols[i] + "_boxplot.jpg"))
        else:
            plt.title(filename)
            fig.savefig(filename+ ".jpg")


        if(visualize):
            plt.show()
        plt.cla()
        plt.close()

def get_correlation_with_attr_categ(dataframe,attributName):
    df_new=dataframe.copy()
    ls_corr_categ_with_attr={}
    df_new[attributName]=pd.cut(dataframe[attributName], 5)
    # print("value counts=\n",df_new[attributName].value_counts())
    # print("df_new[price]=\n",df_new[attributName])
    ##get all category attributes:
    cols = list(dataframe.select_dtypes([object]).columns) ##name of all nominal attributes
    for colName in cols:
        print("colName=",colName)
        chi_out=chisq_of_df_cols(df_new, attributName, colName)
        ls_corr_categ_with_attr[colName]=[chi_out[0]]
    # print("ls_corr_categ_with_attr=",ls_corr_categ_with_attr)
    corr_categ_df = pd.DataFrame.from_dict(ls_corr_categ_with_attr)
    return corr_categ_df

def get_correlation_matrix(df,visual=True,filename=""):
    correlation_matrix=df.corr(method='pearson')
    plt.figure(figsize=[18.8, 18.8],dpi=300)
    sn.heatmap(correlation_matrix, annot=True)
    if(not(filename=="")):
        foldername=filename.split("/")[-2]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        plt.savefig(filename+ ".jpg")

    if(visual):
        plt.show()

    plt.cla()
    plt.close()

    return correlation_matrix


def correlation_with_output(df,ls_attr,visual=True,filename=""):
    plt.figure(figsize=[12.8, 8.8], dpi=300)
    index_names={}
    out_corr=[]
    for i,attr in enumerate(ls_attr):
        corr=get_correlation_with_attr_categ(df, attr)
        out_corr.append(corr)
        index_names[i]=attr
        # print("ManiCorrelation=",ManiCorrelation)
        # EngCorrelation = get_correlation_with_attr_categ(df, "EngineRPM")
        # MassCorrelation = get_correlation_with_attr_categ(df, "MassAirFlow")
    sn.heatmap(pd.concat(out_corr,axis=0,ignore_index=True).rename(index=index_names),annot=True)
    if(not(filename=="")):
        foldername=filename.split("/")[-2]
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        plt.savefig(filename+ ".jpg")

    if(visual):
        plt.show()

    plt.cla()
    plt.close()

def normalize_numeric_attributes(df,method="z-score"):
    ##select all numeric attributes
    numericAttrNames = df.select_dtypes(include=np.number).columns.tolist()
    if(not(method in ["z-score","min-max"])):
        print("Normalization method is undefined. pleas, choose one of this methods ['min-max', 'z-score']")
        return df
    df_new=df.copy()
    if(method=="z-score"):
        for attributeName in numericAttrNames:
            mean = df[attributeName].mean(skipna=True)
            std = df[attributeName].std(skipna=True)
            if(std==0):
                std=0.0000000001
            df_new[attributeName] = (df[attributeName] - mean) /std


    else: ##min-max method
        for attributeName in numericAttrNames:
            min = df[attributeName].min(skipna=True)
            max = df[attributeName].max(skipna=True)
            rg = max - min
            if(rg==0):
                rg=0.0000000001
            df_new[attributeName] = (df[attributeName] - min) /rg

    return df_new


def preprocess_dataset(df):
    print("df.iloc[0,:]=",df.iloc[0,:])
    num_rows=len(df.index)
    # min_null_cnt=15
    # index_min_null=-1
    # max_null_cnt=0
    # index_max_null = -1
    del_indexes=[]
    threshold=5
    for i in range(num_rows):
        cnt=df.iloc[i,:].isna().sum()
        if(cnt>threshold):
            del_indexes.append(i)
        # if(cnt<min_null_cnt):
        #     min_null_cnt=cnt
        #     index_min_null=i
        # if(cnt>max_null_cnt):
        #     max_null_cnt=cnt
        #     index_max_null=i
    print("del_indexes=",del_indexes)
    update_df = df.drop(del_indexes)
    numeric_attribute_name=update_df.select_dtypes(include=np.number).columns.tolist()
    # print("numeric_attribute_name=",numeric_attribute_name)
    # try_filling_null_methods(update_df,"VehicleSpeedInstantaneous")
    for attrName in numeric_attribute_name:
        update_df=apply_fill_missing_method(update_df, attrName, instruction="mean_group",\
                                            num_vals_mean=100)
    #feature selection
    # update_df = cap_data(update_df)
    # get_correlation_matrix(update_df, visual=True, filename="out_figures/correlation_matrix")
    # correlation_with_output(update_df,["ManifoldAbsolutePressure","EngineRPM","MassAirFlow"], visual=True,\
    #                         filename="out_figures/manEngMass_out")
    #
    # correlation_with_output(update_df,["VehicleSpeedInstantaneous","VehicleSpeedAverage"], visual=True,\
    #                         filename="out_figures/instAvgSpeed_out")
    #
    # correlation_with_output(update_df,["LongitudinalAcceleration","VerticalAcceleration"], visual=True,\
    #                         filename="out_figures/longVertAcc_out")

    # correlation_with_output(update_df,["VehicleSpeedAverage","EngineRPM"], visual=True,\
    #                         filename="out_figures/AvgSpeedRPM_out")
    update_df=update_df.drop(["ManifoldAbsolutePressure","MassAirFlow","VehicleSpeedInstantaneous","VerticalAcceleration"],axis=1)
    # cols = list(update_df.select_dtypes([np.int64, np.float64]).columns)
    # draw_box_plots(cols, update_df, foldername="out_figures")
    ##########################################
    ###remove outliers

    update_df = cap_data(update_df)
    # get_correlation_matrix(update_df, visual=True, filename="out_figures/correlation_matrix_after_feature_reduction_outliers")
    #####################
    ##normalize data
    # print("update_df[0]  before norm:", update_df[numeric_attribute_name[3]])
    update_df=normalize_numeric_attributes(update_df,method="z-score")
    # print("update_df[0]  after norm:", update_df[numeric_attribute_name[3]])

    return update_df






def save_dataframe_excel(df,filename):
    foldername=filename.split("/")[-2]
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    df.to_csv(filename + ".csv", index=None)


############################################################
def main():
    data_file_names = [
        'opel_corsa',
        'peugeot'
    ]
    all_data_files={}

    for file_name in data_file_names:
        file_mask = f'dataset/{file_name}_*.csv'
        all_filenames = [i for i in glob.glob(file_mask)]
        all_data_files[file_name] = all_filenames

    print(all_data_files)

    opel_df=get_df_from_csv_files(all_data_files['opel_corsa'])
    peugot_df=get_df_from_csv_files(all_data_files['peugeot'])


    # print("opel_df=",opel_df.columns)
    # print("opel row_number",len(opel_df.index))
    # print(opel_df["AltitudeVariation"])
    # null_index=opel_df[opel_df["drivingStyle"].isnull()].index.tolist()
    # print("indexes:",null_index)
    # print("#################################")
    #
    # print("peugot row_number",len(peugot_df.index))
    # print(peugot_df["AltitudeVariation"])
    # print("opel_df=",peugot_df.columns)
    # print("row_number",len(peugot_df.index))
    # print(peugot_df.loc[0:3,"drivingStyle"])
    # null_index=peugot_df[peugot_df["drivingStyle"].isnull()].index.tolist()
    # print("indexes:",null_index)


    total_df=opel_df.append(peugot_df, ignore_index=True)
    print("total df=",total_df["traffic"])
    total_df=preprocess_dataset(total_df)
    save_dataframe_excel(total_df,"preprocess_dataset/dataset")


if __name__=="__main__":
    main()