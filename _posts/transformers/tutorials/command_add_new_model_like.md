# 通过command 快速添加新模型

## 具体实现

- AddNewModelLikeCommand
    - 函数
        - register_subcommand 注册参数
            - 参数
                - parser: add-new-model-like
                - --config_file: A file with all the information for this model creation.
                - --path_to_repo: When not using an editable install, the path to the Transformers repo.
        - init 初始化
            -  逻辑
                - 读取config_file 参数 并初始化 成员变量
                    - old_model_type
                    - new_model_patterns
                    - add_copied_from
                    - frameworks
                    - old_checkpoint
        - get_user_input
            -  逻辑
                - 
        - create_new_model_like
            - 逻辑 创建一个transformers提供的类似模型
                - 获取目标模型的 各种类
                - get_model_files 获取目标模型的各种文件
                - 校验获取的文件 是否跟框架一致
                - 
        - get_model_files
            - 逻辑  获取目标模型的 doc_file model_files module_name test_files
            - 实现
        - retrieve_model_classes
            - 逻辑  获取model classes
