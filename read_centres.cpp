#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main() {
    // Open the centre_of_boxes.txt file from /dev/shm
    std::ifstream file("/dev/shm/centre_of_boxes.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open /dev/shm/centre_of_boxes.txt" << std::endl;
        return 1;
    }

    std::string line;
    std::cout << "Centre of consistent boxes:" << std::endl;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue; // skip empty lines
        }

        // Remove unwanted characters: '(' , ')' and spaces if needed.
        line.erase(std::remove(line.begin(), line.end(), '('), line.end());
        line.erase(std::remove(line.begin(), line.end(), ')'), line.end());

        // Use a stringstream to extract the two numbers (separated by a comma)
        std::stringstream ss(line);
        double x, y;
        char comma;
        ss >> x >> comma >> y;
        if (ss.fail()) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }
        std::cout << "Centre: (" << x << ", " << y << ")" << std::endl;
    }

    file.close();
    return 0;
} 