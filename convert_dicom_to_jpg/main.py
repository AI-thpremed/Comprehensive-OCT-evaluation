import os
from pydicom import dcmread
from oct_converter.readers import ZEISSDicom
from pypinyin import lazy_pinyin
#
# def convert_to_pinyin_or_keep_original(text):
#     # 检查是否包含中文字符
#     if re.search(r'[\u4e00-\u9fff]', text):
#         # 如果包含中文，转换为拼音
#         return ' '.join(lazy_pinyin(text))
#     else:
#         # 如果不包含中文，保留原内容
#         return text

def search_files(dir_path):
    file_list = os.listdir(dir_path)
    for file_name in file_list:
        complete_file_name = os.path.join(dir_path, file_name)
        if os.path.isdir(complete_file_name):
            search_files(complete_file_name)
        if os.path.isfile(complete_file_name):
            try:
                if(file_name.endswith(".DCM") or file_name.endswith(".dcm")):
                    ds=dcmread(complete_file_name)
                    PatientID = getattr(ds, 'PatientID')
                    PatientSex = getattr(ds, 'PatientSex')
                    PatientBirthDate = getattr(ds, 'PatientBirthDate')
                    Laterality = getattr(ds, 'Laterality')
                    PerformedProcedureStepStartDate = getattr(ds, 'PerformedProcedureStepStartDate')
                    patientid=""



                    for ch in iter(PatientID[4:]):
                        if(ch.isdigit()==True):
                            patientid += str(ch)
                        else:
                            break
                    save_p = r"F:/24_pre_data/sdzy/"+str(PatientID[0:4])+patientid+"/"
                    if (os.path.exists(save_p) == False):
                        os.mkdir(save_p)

                    img = ZEISSDicom(complete_file_name)
                    oct_volumes, fundus_volumes = img.read_data()

                    if(len(oct_volumes)>0):

                        if (oct_volumes[0].num_slices > 1):
                            save_path = save_p + str(PatientID[0:4]) + patientid + "_" + file_name + "/"
                            if (os.path.exists(save_path) == False):
                                os.mkdir(save_path)
                            for idx, volume in enumerate(oct_volumes):
                                volume.save(save_path + str(PatientID[0:4])+patientid + "_" + str(PatientSex[0:1]) + "_" + str(PatientBirthDate[0:8]) + "_" + str(Laterality[0:2]) + "_" + str(PerformedProcedureStepStartDate[0:8]) + ".jpg")

                    if(len(fundus_volumes)>0):

                        if (fundus_volumes[0].num_slices > 1):
                            save_path1 = save_p + str(PatientID[0:4]) + patientid + "_" + file_name + "/"
                            if (os.path.exists(save_path1) == False):
                                os.mkdir(save_path1)
                            for idx, image in enumerate(fundus_volumes):

                                # filename = str(PatientID[0:4]) + patientid + "_" + str(PatientSex[0:1]) + "_" + str(
                                #     PatientBirthDate[0:8]) + "_" + str(Laterality[0:2]) + "_" + str(
                                #     PerformedProcedureStepStartDate[0:8]) + ".jpg"
                                #
                                # # 将字符串转换为UTF-8编码
                                # filename = filename.encode("utf-8")
                                #
                                # # 保存图片
                                # image.save(os.path.join(save_path, filename))

                                image.save(save_path1 + str(PatientID[0:4])+patientid + "_" + str(PatientSex[0:1]) + "_" + str(PatientBirthDate[0:8]) + "_" + str(Laterality[0:2]) + "_" + str(PerformedProcedureStepStartDate[0:8]) + ".jpg")

            finally:
                continue


if __name__ == '__main__':
    dir_path = r"F:\24年前瞻性其他的外部数据\深大oct\215507186057331-E-20240816150231\DataFiles"

    search_files(dir_path)

    print("解析完成")
