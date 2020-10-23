

#include "nnet-io.h"

namespace DNN{
    
    int Peek(std::istream &is, bool binary) {
        if (!binary) is >> std::ws;  // eat up whitespace.
        return is.peek();
    }
    uint32_t ReadUint32(std::istream &is) {
        union{
            char res[4];
            uint32_t ans;
        }u;
        is.read(u.res, 4);
        if(is.fail()){
            LOG_ERROR("Error ReadUint32");
        }
        return u.ans;
    }
    void WriteUint32(std::ostream &os, uint32_t i){
        union{
            char buf[4];
            uint32_t ans;
        }u;
        u.ans = i;
        os.write(u.buf, 4);
        if(os.fail()){
            LOG_ERROR("Error WriteUint32");
        }
    }
    void ExpectToken(std::istream &is, bool binary, const std::string & token) {
        ExpectToken(is, binary, token.c_str());
    }
    void ExpectToken(std::istream &is, bool binary, const char *token) {
        int pos_at_start = is.tellg();
        LOG_ASSERT(token != NULL);
        if (!binary) is >> std::ws;  // consume whitespace.
        std::string str;
        is >> str;
        is.get();  // consume the space.
        if (is.fail()) {
            LOG_ERROR("Failed to read token [started at file position "
                      << pos_at_start << "], expected " << token);
        }
        if (strcmp(str.c_str(), token) != 0) {
            LOG_ERROR("Expected token \"" << token << "\", got instead \""
                      << str <<"\".");
        }
    }
    void ReadToken(std::istream &is, bool binary, std::string *str) {
        LOG_ASSERT(str != NULL);
        if (!binary) is >> std::ws;  // consume whitespace.
        is >> *str;
        if (is.fail()) {
            LOG_ERROR("ReadToken, failed to read token at file position "
                      << is.tellg());
        }
        if (!isspace(is.peek())) {
            LOG_ERROR("ReadToken, expected space after token, saw instead "
                      << static_cast<char>(is.peek())
                      << ", at file position " << is.tellg());
        }
        is.get();  // consume the space.
    }
    void WriteToken(std::ostream &os, bool binary, const std::string & token) {
        // binary mode is ignored;
        // we use space as termination character in either case.
        os << token << " ";
        if (os.fail()) {
            LOG_ERROR("Write failure in WriteToken.");
        }
    }
    
}

