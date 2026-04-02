#include <arg.h>
#include <common.h>
#include <llama.h>
#include <mtmd.h> //视觉必须
#include<mtmd-helper.h>
#include<iostream>
#include<filesystem>
#include <algorithm>
#include<vector>
#include<fstream>

using namespace std;
namespace fs = std::filesystem;

std::string _INPUT = "inputImg";
const char* QWEN3VLMDL = "assets\\Qwen3VL-4B-Instruct-Q8_0.gguf";
const char* MMPROJ = "assets\\mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf";
const char* PROMPT = "你是一位LoRA打标专家，你的任务是根据我给定的图片，客观描述画面内容，你只需要输出纯净的结果即可。请关注以下方面：环境细节、人物细节、面部特征、各种服饰、人物动作、外观细节、表情、姿势动作、环境场景背景地点、时间、光照条件、风格与材质。禁止出现抽象名词形容词，忽略图片中的任何水印。输出时仅使用纯文本格式，禁止包含除纯文本外的任何文档格式，禁止出现换行符、制表符等符号，禁止出现换行符、制表符等符号。最终形成一段纯中文的自然语言提示词。";
const int MAX_GEN = 8192;

bool isImageFile(string path){
    static const std::vector<std::string> imageExtensions = {
        ".jpg", ".jpeg", ".png", ".bmp"
    };
    std::string ext = path.substr(path.find_last_of("."));
    
    // 转换为小写进行比较
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    return std::find(imageExtensions.begin(), 
                     imageExtensions.end(), 
                     ext) != imageExtensions.end();
}


void getImageFile(vector<string> &fl){
try {
        // 检查目录是否存在
        if (!fs::exists(_INPUT) || !fs::is_directory(_INPUT)) {
            std::cerr << "目录不存在或不是有效目录: " << _INPUT << std::endl;
            return;
        }
        
        // 遍历目录
        for (const auto& entry : fs::directory_iterator(_INPUT)) {
            if (fs::is_regular_file(entry.status())) {
                std::string filename = entry.path().filename().string();
                
                if (isImageFile(filename)) {
                    // 获取完整路径
                    std::string fullPath = entry.path().string();
                    fl.push_back(fs::absolute(fullPath).string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "文件系统错误: " << e.what() << std::endl;
    }
    
    return;
}

int main(int argc, char const *argv[])
{
    //文件初始化
    std::string trigger = "My_lora,";
    if(argc >= 3) {
        _INPUT = argv[2];
        trigger = argv[1];
        trigger += ",";
    }
    std::cout << "Lora_name:" << trigger << " input_name:" << _INPUT << "\n";
    // = argc >= 2 ? (std::string(argv[argc-1]) + u8",") : "";
    vector<string> FileList;
    getImageFile(FileList);
    //vlm初始化
    llama_backend_init();
    common_params params;
    params.model.path = fs::absolute(QWEN3VLMDL).string();
    params.mmproj.path = fs::absolute(MMPROJ).string();
    params.prompt = u8"<__image__>\n" + std::string(PROMPT);
    params.n_predict = 8192;    //最大生成长度
    
    int n_predict = params.n_predict;
    llama_numa_init(params.numa);

    //手动初始化后端
    ggml_backend_reg_t reg = ggml_backend_load("ggml-cuda.dll");
    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    if (!(reg && cpu_reg)){std::cerr << "ggml_backend_t failed";}
    ggml_backend_register(reg);
    ggml_backend_register(cpu_reg);

    /*=============================初始化结束====================================*/

    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    mtmd_context *mtmd = nullptr;
    mtmd_input_chunks *chunks = nullptr;
    mtmd_bitmap *bitmap = nullptr;
    const llama_vocab *vocab = nullptr;

    llama_sampler_chain_params sparams = {};
    llama_sampler *sampler_chain = nullptr;
    
    mtmd_context_params mtmd_params = {};
    mtmd_input_text input_text = {};
    const mtmd_bitmap *bitmaps[1] = { nullptr };
    std::vector<llama_token> tokens;

    // Load LLM

    llama_model_params model_params = common_model_params_to_llama(params);
    model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (model == NULL) {
        std::cout << ": error: unable to load model\n";
        return -1;
    }
    vocab = llama_model_get_vocab(model);
    if (vocab == NULL) {
        std::cout << "failed to get vocab\n";
        return -1;
    }

    //Create ctx
    const int n_kv_req = 8192;
    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.n_ctx   = n_kv_req;
    ctx_params.n_batch = 4096;  //单次加载图像token数
    ctx = llama_init_from_model(model,ctx_params);
    if (ctx == NULL) {
        std::cout << "create context failed\n";
        return -1;
    }

    //Simple chain
    sparams = llama_sampler_chain_default_params();
    sampler_chain = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k(5));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_penalties(64, 1.2f, 0.1f, 0.1f));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_dist(time(NULL) ^ 0xABCD));

    //MTMD support
    mtmd_params = mtmd_context_params_default();
    mtmd_params.print_timings = false;

    mtmd = mtmd_init_from_file(MMPROJ,model,mtmd_params);
    if (mtmd == NULL) {
        std::cout << "mtmd_init_from_file failed\n";
        return -1;
    }
    for(auto &image_path:FileList){
        auto seed = time(NULL);
        std::cout << "seed:" << (seed^0xABCD) << "\n";
        ctx = llama_init_from_model(model,ctx_params);
        if (ctx == NULL) {
            std::cout << "create context failed\n";
            return -1;
        }
        const llama_vocab *vocab = llama_model_get_vocab(model);
        if (vocab == NULL) {
            std::cout << "failed to get vocab\n";
            return -1;
        }
        llama_sampler *sampler_chain = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k(5));
        //llama_sampler_chain_add(sampler_chain, llama_sampler_init_penalties(64, 1.2f, 0.1f, 0.1f));
        llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp(0.5f));
        llama_sampler_chain_add(sampler_chain, llama_sampler_init_penalties(96, 1.2f, 0.2f, 0.2f));
        llama_sampler_chain_add(sampler_chain, llama_sampler_init_dist(seed));
        mtmd_bitmap *bitmap = mtmd_helper_bitmap_init_from_file(mtmd, image_path.c_str());
        mtmd_input_chunks *chunks = mtmd_input_chunks_init();
        if (chunks == NULL) {
            std::cout << "mtmd_input_chunks_init failed\n";
            return -1;
        }
        mtmd_input_text input_text = {};
        input_text.text = params.prompt.c_str();
        input_text.add_special = true;
        input_text.parse_special = true;
        bitmaps[0] = bitmap;
        int32_t tok_res = mtmd_tokenize(mtmd, chunks, &input_text, bitmaps, 1);
        llama_pos n_past = 0;
        llama_seq_id seq_id = 0;
        if (mtmd_helper_eval_chunks(mtmd, ctx, chunks, n_past, seq_id, n_kv_req, true, &n_past)){
            std::cout << "mtmd_helper_eval_chunks failed\n";
            return -1;
        }

        std::string response;
        int emptyLine_brk = 0;
        // Generate loop
        for (int i = 0; i < MAX_GEN; ++i) {
            if (n_past >= llama_n_ctx(ctx)) {
                std::cout << "Context full";
                break;
            }

            llama_token next = llama_sampler_sample(sampler_chain, ctx, -1);
            llama_sampler_accept(sampler_chain, next);

            if (next == llama_vocab_eos(vocab) || next == llama_vocab_eot(vocab) || next == llama_vocab_nl(vocab)) {
                std::cout << ("EOS/EOT/NL reached");
                break;
            }

            char piece[128] = {0};
            int len = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, true);

            if (len > 0 && len < (int)sizeof(piece)) {
                std::string token_str(piece, len);
                if(token_str.find("\n") != std::string::npos){
                    response += token_str;
                    emptyLine_brk++;
                    if(emptyLine_brk > 2) break;
                }
                response += token_str;
            }
            // Decode next token
            llama_batch batch = llama_batch_init(1, 0, 1);
            if (!batch.token) {
                std::cout << ("batch alloc failed in loop");
                break;
            }

            batch.n_tokens = 1;
            batch.token[0] = next;
            batch.pos[0] = n_past++;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = seq_id;
            batch.logits[0] = 1;

            int decode_res = llama_decode(ctx, batch);
            llama_batch_free(batch);

            if (decode_res != 0) {
                std::cout << ("llama_decode failed: %d", decode_res);
                break;
            }
        }
        fstream fs;
        //auto i = FileList[0];
        fs.open([image_path]()->std::string{
            auto str = image_path;
            str.replace((std::size_t)image_path.find_last_of('.'),(std::size_t)image_path.size()-1,".txt");
            return str;
        }(),ios::out);
        if(fs.is_open()) fs << trigger << response;
        fs.close();
        llama_sampler_free(sampler_chain);
        mtmd_bitmap_free(bitmap);
        mtmd_input_chunks_free(chunks);
        llama_free(ctx); 
        std::cout << "\n";
    }
    /*======================================清理资源=====================================*/
    mtmd_free(mtmd);      // 如果用了视觉模块
    llama_model_free(model);    // model weights
    ggml_backend_unload(cpu_reg);
    ggml_backend_unload(reg);
    llama_backend_free();
    return 0;
}



