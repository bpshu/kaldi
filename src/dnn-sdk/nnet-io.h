

#ifndef nnet_io_h
#define nnet_io_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "nnet-comm.h"

namespace DNN{
    int Peek(std::istream &is, bool binary);
    uint32_t ReadUint32(std::istream &is);
    void WriteUint32(std::ostream &os, uint32_t i);
    void ReadToken(std::istream &is, bool binary, std::string *str);
    void WriteToken(std::ostream &os, bool binary, const std::string & token);
    void ExpectToken(std::istream &is, bool binary, const char *token);
    void ExpectToken(std::istream &is, bool binary, const std::string & token);
    template<typename T>
    void WriteBasicType(std::ostream &os, bool binary, T f) {
        if (binary) {
            char c = sizeof(f);
            os.put(c);
            os.write(reinterpret_cast<const char *>(&f), sizeof(f));
        } else {
            os << f << " ";
        }
    }
    template<typename T>
    void ReadBasicType(std::istream &is, bool binary, T *f) {
        if (binary) {
            double d;
            int c = is.peek();
            if (c == sizeof(*f)) {
                is.get();
                is.read(reinterpret_cast<char*>(f), sizeof(*f));
            } else if (c == sizeof(d)) {
                ReadBasicType(is, binary, &d);
                *f = d;
            } else {
                LOG_ERROR("ReadBasicType: expected float, saw " << is.peek()
                          << ", at file position " << is.tellg());
            }
        } else {
            is >> *f;
        }
        if (is.fail()) {
            LOG_ERROR("ReadBasicType: failed to read, at file position "
                      << is.tellg());
        }
    }

}

#endif /* nnet_io_h */
