#include "tools/convert.hpp"

namespace kurff{
    void Convert::init(const string& name, const string& backend){
        db_.reset(db::GetDB(backend));
        db_->Open(name, db::NEW);
        txn_.reset(db_->NewTransaction());

    }
}