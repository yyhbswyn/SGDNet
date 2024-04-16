import pickle as pkl
file = open('/home/hjj/wuyini_pro/mmaction2/mmaction/datasets/sentence_demo.pkl','rb')
results = pkl.load(file)
age_group = ['1:adult','2:child','3:senior']
base_action = ['4:walk','5:stand','6:run','7:sit','8:bow','9:crouch']
traffic_action=['10:take the zebra-crossing','11:jaywalk','12:wait for passing','13:walk on the roadside','14:ride bike','15:ride motorcycle',
                                '16:hand wave','17:deliver goods','18:push stroller or wheelchair','19:get on two wheeled vehicle','20:get off two wheeled vehicle',
                                '21:get on four wheeled vehicle','22:get off four wheeled vehicle']
non_traffic_action=['23:talk with others','24:look at phone or something else', '25:talk on the phone','26:enter into building',
                                    '27:exit from the building','28:load','29:unload','30:open the door','31:close the door']

all_class_info = age_group+base_action+traffic_action+non_traffic_action

text_aug=[f"this person's age group belongs to {{}}, the actions are {{}}"]

# for i in range(len(data['entity_ids'])):  #这个i代表的是人的个数。data['entity_ids'][i]就是对应的人的id
#     print(i)


def gen_person_sentence(results):
    age_group = ['1:adult', '2:child', '3:senior']
    base_action = ['4:walk', '5:stand', '6:run', '7:sit', '8:bow', '9:crouch']
    traffic_action = ['10:take the zebra-crossing', '11:jaywalk', '12:wait for passing', '13:walk on the roadside',
                      '14:ride bike', '15:ride motorcycle',
                      '16:hand wave', '17:deliver goods', '18:push stroller or wheelchair',
                      '19:get on two wheeled vehicle', '20:get off two wheeled vehicle',
                      '21:get on four wheeled vehicle', '22:get off four wheeled vehicle']
    non_traffic_action = ['23:talk with others', '24:look at phone or something else', '25:talk on the phone',
                          '26:enter into building',
                          '27:exit from the building', '28:load', '29:unload', '30:open the door', '31:close the door']

    all_class_info = age_group + base_action + traffic_action + non_traffic_action

    text_aug = [f"person's id is {{}},the age group belongs to {{}}, the actions are {{}}"]
    for i in range(len(results['entity_ids'])):  # 这个i代表的是人的个数。data['entity_ids'][i]就是对应的人的id,一共11个人的话，0，1，···10
        person_nature_list=[]   #每个人都给一个列表，存储自然语言表示的年龄组、行为标签。
        sentence=[]
        person_all_label = results['gt_labels'][i][1:]  #这样是获取这个人的真正的label信息,一共31列，0-31,[1,0,0,1,0,0] person_all_label[x]=1代表
        for index in range(len(all_class_info)):  #index=0-30,索引数正好比类别id数少1.
            action_id = all_class_info[index].split(':')[0]  #action_id是获取上面我定义好的行为类别的id，是从1开始的。1=adult,2=child
            action_nature = all_class_info[index].split(':')[1]  #这是获取类别的自然语言描述信息
            if(person_all_label[index]==1):   #index从0开始，person_all_label[index]==1,说明对应到上面的类别,要把类别映射的id+1的类别替换成自然语言。
                person_nature_list.append(action_nature)
        # print(person_nature_list)   #['adult', 'walk', 'walk on the roadside']
        # 对一个人的list而言，只有第一个代表的是年龄组，剩下的就是他真正的行为信息
        for ii, txt in enumerate(text_aug):  # 开始填充句子，从0开始。有几条数据，填几条对应信息的句子。
            sentence.append(txt.format(results['entity_ids'][i], person_nature_list[0], (',').join(person_nature_list[1:])))



gen_person_sentence(results)



