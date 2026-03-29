#include<vector>
#include<tuple>
#include<unordered_map>
#include<set>
#include<numeric>
#include<algorithm>
#include<variant>
#include<filesystem>
#include<opencv2/opencv.hpp>
#include<execution>

class KeyFrameSelector{
private:
    // 权重设置
    std::vector<double>sharpness_weights{0.33, 0.33, 0.33};
    std::vector<double>visual_weights{0.3, 0.3, 0.4};
    std::vector<double>final_weights{0.45, 0.55};
    
    // 阈值设置 
    double top_sharpness_percent = 0.35;
    double top_visual_percent = 0.45;
    double similarity_threshold = 0.990;

    int histSize = 256;
    float range[2] = {0.0f, 256.0f};  // 范围
    const float* histRange = {range};
public:
    KeyFrameSelector(/* args */) = default;
    ~KeyFrameSelector() = default;
    std::vector<std::tuple<std::string,int,int>> extract_video_segments(std::string video_path,int segment_duration = 15){
        std::cout << video_path << "\n";
        cv::VideoCapture cap = cv::VideoCapture(video_path);
        if(!cap.isOpened()){
            throw std::runtime_error("无法打开视频文件" + video_path);
        }
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = (long)cap.get(cv::CAP_PROP_FRAME_COUNT);
        double duration = total_frames / fps;
        std::vector<std::tuple<std::string,int,int>> segments;
        if(duration <= segment_duration){
            segments.push_back({video_path, 0, total_frames});
        }else{
            int segment_frames = (int)(segment_duration * fps);
            for (int start_frame = 0; start_frame < total_frames; start_frame += segment_frames) {
                int end_frame = std::min(start_frame + segment_frames, total_frames);
                segments.push_back({video_path, start_frame, end_frame});
            }

        }
        cap.release();
        return segments;
    }

    std::tuple<double,double,double> calculate_sharpness_metrics(cv::Mat &frame){
        cv::Mat gray;
        if(frame.channels() == 3){
            cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
        }else{
            gray = frame;
        }
        cv::Mat laplacian;
        cv::Laplacian(gray,laplacian,CV_64F);
        double laplacian_var = cv::mean(laplacian.mul(laplacian))[0]; // 方差计算
        
        // Tenengrad梯度
        cv::Mat sobelx, sobely;
        cv::Sobel(gray, sobelx, CV_64F, 1, 0, 3);
        cv::Sobel(gray, sobely, CV_64F, 0, 1, 3);
        cv::Mat tenengrad_mat = sobelx.mul(sobelx) + sobely.mul(sobely);
        double tenengrad = cv::mean(tenengrad_mat)[0];
        
        // Brenner梯度
        cv::Mat brenner_mat;
        cv::subtract(gray(cv::Range(2, gray.rows), cv::Range::all()),
                    gray(cv::Range(0, gray.rows-2), cv::Range::all()),
                    brenner_mat, cv::noArray(), CV_64F);
        brenner_mat = brenner_mat.mul(brenner_mat);
        double brenner = cv::mean(brenner_mat)[0];
        
        return {laplacian_var, tenengrad, brenner};
    }

    std::tuple<double, double, double> calculate_visual_metrics(const cv::Mat& frame){
        cv::Mat gray;
    
        // 转换为灰度图
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = frame.clone();
        }
        
        // 亮度与对比度
        cv::Scalar mean_brightness = cv::mean(gray);
        double brightness = mean_brightness[0];
        
        cv::Mat mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        double contrast = stddev.at<double>(0, 0);
        double brightness_contrast_score = brightness * 0.5 + contrast * 0.5;
        
        // 色彩饱和度
        double saturation = 0.0;
        if (frame.channels() == 3) {
            cv::Mat hsv;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            std::vector<cv::Mat> hsv_channels;
            cv::split(hsv, hsv_channels);
            cv::Scalar mean_saturation = cv::mean(hsv_channels[1]);
            saturation = mean_saturation[0];
        }
        
        // 模糊检测（拉普拉斯方差）
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar mean_laplacian, stddev_laplacian;
        cv::meanStdDev(laplacian, mean_laplacian, stddev_laplacian);
        double blur_score = stddev_laplacian[0] * stddev_laplacian[0]; // 方差 = 标准差^2
        
        return {brightness_contrast_score, saturation, blur_score};
    }

    std::vector<double> normalize_scores(std::vector<double> &scores){
        if(scores.empty()) return {};
        auto [min_it, max_it] = std::minmax_element(scores.begin(), scores.end());
        double min_val = *min_it;
        double max_val = *max_it;
        if (max_val == min_val) {
            return std::vector<double>(scores.size(),0.5);
        }
        std::vector<double> normalized;
        normalized.reserve(scores.size());
        for (double score : scores) {
                normalized.push_back(((score - min_val)*1.0) / (max_val - min_val));
        }
        return normalized;
    }

    std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> process_segment_sharpness(std::tuple<std::string,int,int> &segment_info){
        auto [video_path, start_frame, end_frame] = segment_info;
        cv::VideoCapture cap = cv::VideoCapture(video_path);
        cap.set(cv::CAP_PROP_POS_FRAMES,start_frame);
        std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> frames_data;
        std::vector<double> laplacian_scores,
            tenengrad_scores,
            brenner_scores;
        std::vector<int> frame_positions;
        int current_frame = start_frame;
        while (current_frame < end_frame){
            cv::Mat frame;
            bool ret = cap.read(frame);
            if(!ret) break;
            auto [laplacian, tenengrad, brenner] = this->calculate_sharpness_metrics(frame);
            frames_data.push_back({
                {"frame", std::move(frame)},
                {"position", current_frame},
                {"laplacian", laplacian},
                {"tenengrad", tenengrad},
                {"brenner", brenner}});
            laplacian_scores.push_back(laplacian);
            tenengrad_scores.push_back(tenengrad);
            brenner_scores.push_back(brenner);
            frame_positions.push_back(current_frame);
            
            current_frame += 1;
            /* code */
        }
        cap.release();
        if(frames_data.empty()) return{};
        std::vector<double> norm_laplacian = normalize_scores(laplacian_scores);
        std::vector<double> norm_tenengrad = normalize_scores(tenengrad_scores);
        std::vector<double> norm_brenner = normalize_scores(brenner_scores);
        std::vector<double> sharpness_scores;
        for(int i = 0;i < frames_data.size();i++){
            double weighted_score = (norm_laplacian[i] * this->sharpness_weights[0] + 
                norm_tenengrad[i] * this->sharpness_weights[1] + 
                norm_brenner[i] * this->sharpness_weights[2]);
            frames_data[i].insert({
                {"norm_laplacian", norm_laplacian[i]},
                {"norm_tenengrad", norm_tenengrad[i]},
                {"norm_brenner", norm_brenner[i]},
                {"sharpness_score", weighted_score}
            });
            sharpness_scores.push_back(weighted_score);
        }

        int num_select = std::max(1,static_cast<int>(frames_data.size()*this->top_sharpness_percent));
        std::vector<size_t> indices(sharpness_scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&sharpness_scores](size_t i1, size_t i2) {
                return sharpness_scores[i1] > sharpness_scores[i2];
            });
        
        std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> selected_frames;
        for (int i = 0; i < num_select; i++) {
            selected_frames.push_back(frames_data[indices[i]]);
        }
        
        return selected_frames;
    }

    std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> process_visual_quality(std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> &sharpness_frames){
        std::vector<double> brightness_scores,saturation_scores,blur_scores;
        std::for_each(std::execution::par,sharpness_frames.begin(),sharpness_frames.end(),[&](std::unordered_map<std::string, std::variant<int, double, cv::Mat, std::string>> &frame_data){
            auto [brightness, saturation, blur] = this->calculate_visual_metrics(std::get<cv::Mat>(frame_data.at("frame")));
            frame_data.insert({                
                {"brightness", brightness},
                {"saturation", saturation},
                {"blur", blur}}
            );
            brightness_scores.push_back(brightness);
            saturation_scores.push_back(saturation);
            blur_scores.push_back(blur);
        });

        std::vector<double> norm_brightness = normalize_scores(brightness_scores);
        std::vector<double> norm_saturation = normalize_scores(saturation_scores);
        std::vector<double> norm_blur = normalize_scores(blur_scores),visual_scores;
        for(int i = 0;i < sharpness_frames.size();i++){
            double visual_score = (norm_brightness[i] * visual_weights[0] + 
                          norm_saturation[i] * visual_weights[1] + 
                          norm_blur[i] * visual_weights[2]);
            for (const auto& [key, value] : std::unordered_map<std::string, std::variant<int, double, cv::Mat, std::string>>{
                {"norm_brightness", norm_brightness[i]},
                {"norm_saturation", norm_saturation[i]},
                {"norm_blur", norm_blur[i]},
                {"visual_score", visual_score}
            }) {
                sharpness_frames[i][key] = value;  // 新值会覆盖旧值
            }
            visual_scores.push_back(visual_score);
        }

        int num_select = std::max(1,static_cast<int>(sharpness_frames.size()*this->top_visual_percent));
        std::vector<size_t> indices(visual_scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&visual_scores](size_t i1, size_t i2) {
                return visual_scores[i1] > visual_scores[i2];
            });
        
        std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> selected_frames;
        for (int i = 0; i < num_select; i++) {
            selected_frames.push_back(sharpness_frames[indices[i]]);
        }
        return selected_frames;
    }

    double calculate_histogram_similarity(cv::Mat hist1,cv::Mat hist2){
        return cv::compareHist(hist1,hist2,cv::HISTCMP_CORREL);
    }

    std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> process_similarity_segment(std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> &visual_frames){
        if(visual_frames.size() <= 1){
        for(auto &frame_data:visual_frames){
                double final_score = (std::get<double>(frame_data["sharpness_score"]) * final_weights[0] + 
                    std::get<double>(frame_data["visual_score"]) * final_weights[1]);
                frame_data["final_score"] = final_score;
            }
            return visual_frames;
        }

        std::vector<cv::Mat> histograms;
        for(auto &frame_data:visual_frames){
            cv::Mat gray,hist;
            cv::cvtColor(std::get<cv::Mat>(frame_data.at("frame")),gray,cv::COLOR_BGR2GRAY);
            cv::calcHist(&gray,1,0,cv::Mat(),hist,1,&histSize,&histRange);
            cv::normalize(hist,hist,0,1,cv::NORM_MINMAX);
            histograms.push_back(hist);
        }
        std::vector<std::vector<int>>similar_groups;
        std::set<int> used_indices;
        for(int i = 0;i < visual_frames.size();i++){
            if(used_indices.find(i) != used_indices.end()) continue;

            std::vector<int> group = {i};
            used_indices.insert(i);
            for(int j = i+1;j< visual_frames.size();j++){
                if(used_indices.find(j) != used_indices.end()) continue;
                double similarity = this->calculate_histogram_similarity(histograms[i],histograms[j]);
                if(similarity > this->similarity_threshold){
                    group.push_back(j);
                    used_indices.insert(j);
                }
            }

            similar_groups.push_back(group);
        }

        std::vector<std::unordered_map<std::string, std::variant<int, double, cv::Mat,std::string>>> final_frames;
        for(auto &group : similar_groups){
            int best_idx = 0,best_score = -1;
            if(group.size() == 1) best_idx = group[0];
            else{
                best_idx = group[0];
                for(auto &idx : group){
                    auto frame_data = visual_frames[idx];
                    double final_score = (std::get<double>(frame_data["sharpness_score"]) * final_weights[0] + 
                        std::get<double>(frame_data["visual_score"]) * final_weights[1]);
                    if(final_score > best_score){
                        best_score = final_score;
                        best_idx = idx;
                    }
                }
            }

            auto frame_data = visual_frames[best_idx];
            double final_score = (std::get<double>(frame_data["sharpness_score"]) * final_weights[0] + 
                        std::get<double>(frame_data["visual_score"]) * final_weights[1]);
            frame_data["final_score"] = final_score;
            final_frames.push_back(frame_data);
        }

        return final_frames;
    }

    void save_frames_step5(std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> &frames,std::string output_dir){
        for(auto &frame_data:frames){
            std::string filename = std::to_string(std::get<int>(frame_data.at("position")))+"_"+
                std::to_string(std::get<double>(frame_data.at("final_score"))*100)+".png";
            
            std::string filepath = std::filesystem::path(output_dir).append(filename).string();
            cv::imwrite(filepath, std::get<cv::Mat>(frame_data["frame"]));
        }
    }

    std::vector<std::unordered_map<std::string,std::variant<int,double,cv::Mat,std::string>>> select_keyframes(std::string video_path,std::string output_dir = "./output"){
        /*主函数*/
        std::string video_name = std::filesystem::path(video_path).stem().string();
        std::cout << "开始处理视频:" << video_name << "\n";
        
        std::string main_output_dir = std::filesystem::path(output_dir).string();
        // 步骤1: 提取视频片段
        auto segments = this->extract_video_segments(video_path);
        std::cout << "视频分割为" << segments.size() << " 个片段" << "\n";

        // 处理每个片段
        std::vector<std::unordered_map<std::string, std::variant<int, double, cv::Mat, std::string>>> all_final_frames{};
        for(int i = 0;i < segments.size();i++){
            auto segment = segments[i];
            std::cout << "处理片段 " << i + 1 << "/" << segments.size() << ": 帧范围 " << std::get<1>(segment) << "-" << std::get<2>(segment) << "\n";
            try
            {
                auto sharpness_frames = this->process_segment_sharpness(segment);
                if(sharpness_frames.empty()){ 
                    std::cout << "片段 " << i + 1 << " 清晰度筛选无结果" << "\n";
                    continue;
                }
                std::cout << "片段 " << i + 1 << " 清晰度筛选得到 " << sharpness_frames.size() <<" 帧\n";

                auto visual_frames = this->process_visual_quality(sharpness_frames);
                if(visual_frames.empty()){ 
                    std::cout << "片段 " << i + 1 << " 视觉质量筛选无结果\n";
                    continue;
                }
                std::cout << "片段 " << i + 1 << " 视觉质量筛选得到 " << visual_frames.size() <<" 帧\n";

                //相似性筛选
                auto similarity_frames = this->process_similarity_segment(visual_frames);
                std::cout << "片段 " << i + 1 << " 相似性筛选得到 " << similarity_frames.size() <<" 帧\n";

                //保存步骤5结果到output2

                std::string output2_dir = std::filesystem::path(main_output_dir).append("general").string();
                std::filesystem::create_directories(output2_dir);
                this->save_frames_step5(similarity_frames,output2_dir);
                all_final_frames.insert(all_final_frames.end(),similarity_frames.begin(),similarity_frames.end());
            }
            catch(const std::exception& e)
            {
                std::cerr << "处理片段 "<< i+1 << " 时出错: " << e.what() << '\n';
                continue;
            }
        
        }
        std::cout << "所有片段处理完成，共得到" <<  all_final_frames.size() << " 个候选帧\n";
        return all_final_frames;
    }

};

int main(int argc, char const *argv[])
{
    KeyFrameSelector *selector = nullptr;
    std::string video_path = argv[1],
        output_dir = argv[2];

    try{
        selector = new KeyFrameSelector();
        auto final_keyframes = selector->select_keyframes(video_path,output_dir);
        std::cout << "\n处理完成！最终选择了 " << final_keyframes.size() << " 个关键帧\n";// << "结果保存在: " << std::filesystem::path(output_dir).append(video_path).stem().append("opt").string();
        delete selector;
    }catch(const std::exception& e){
        std::cerr <<"处理视频时出错: " << e.what() << '\n';
        if(selector != nullptr) delete selector;
        }
    return 0;
}
