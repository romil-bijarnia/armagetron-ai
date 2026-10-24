#include "../../../src/tron/gLightcycleAsset.h"
#include <fstream>
#include <sstream>
#include <cassert>
#include <iterator>

int main(int argc,char **argv)
{
    assert(argc==2);
    std::ifstream input(argv[1],std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(input)),{});
    gLightcycleAsset::Asset asset;
    std::istringstream valid(bytes);
    assert(gLightcycleAsset::Load(valid,asset));
    std::size_t count=0;
    for(auto const & group:asset.parts) for(auto const & part:group) count+=part.size();
    assert(count==51744*3);
    assert(asset.parts[3][5].size()>60000); // The textured rider is present.
    assert(!asset.parts[1][3].empty() && !asset.parts[2][3].empty());
    auto reject=[&](std::string const & candidate)
    {
        std::istringstream stream(candidate);
        assert(!gLightcycleAsset::Load(stream,asset));
        assert(asset.parts[3][5].size()>60000); // A failed read is transactional.
    };
    reject(bytes.substr(0,bytes.size()-1));
    reject(bytes+"unexpected trailing data");
    std::string corrupt=bytes;corrupt[0]='?';reject(corrupt);
    corrupt=bytes;corrupt[40]=char(0xff);corrupt[41]=char(0xff);corrupt[42]=char(0xff);corrupt[43]=char(0x7f);reject(corrupt);
    // First vertex coordinate is a quiet NaN in little-endian IEEE 754.
    corrupt=bytes;corrupt[44]=0;corrupt[45]=0;corrupt[46]=char(0xc0);corrupt[47]=char(0x7f);reject(corrupt);
}
