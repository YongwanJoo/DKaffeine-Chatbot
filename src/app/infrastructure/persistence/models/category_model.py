"""Category 데이터베이스 모델"""
from sqlalchemy import Column, BigInteger, String, DateTime
from sqlalchemy.sql import func
from app.infrastructure.persistence.models.base import Base


class Category(Base):
    """카테고리 엔티티 (Java 백엔드: com.dkaffein.datasource.domain.entity.Category와 일치)
    
    Java 백엔드 기준:
    - categoryId: Long (PK)
    - name: String (NOT NULL)
    - description: String (NULL)
    - useYn: Boolean (NOT NULL, default true)
    - BaseEntity 상속: createdAt, updatedAt, deletedAt
    """
    __tablename__ = "category"
    
    # Java: @Id @GeneratedValue, 컬럼명: category_id
    id = Column("category_id", BigInteger, primary_key=True, autoincrement=True)
    
    # Java: name (String, NOT NULL)
    name = Column("name", String(255), nullable=False)
    
    # Java: description (String, NULL)
    description = Column("description", String(1000), nullable=True)
    
    # Java: useYn (Boolean, NOT NULL) - Python에서는 Boolean 대신 Integer(0/1)나 String('Y'/'N')을 쓸 수도 있으나, 
    # SQLAlchemy Boolean은 DB에 따라 적절히 매핑됨 (PostgreSQL: BOOLEAN, MySQL: TINYINT)
    # 여기서는 Java와 맞추기 위해 Boolean 사용 (또는 String('Y'/'N') 확인 필요, 일단 Boolean 가정)
    # 만약 Java가 char(1) 'Y'/'N'을 쓴다면 String(1)로 변경해야 함.
    # 일단 안전하게 String(1)로 하고 default 'Y'로 설정 (일반적인 레거시 패턴)
    # 혹은 Boolean으로 하고 True/False 매핑.
    # 기존 코드 스타일을 따름.
    use_yn = Column("use_yn", String(1), nullable=False, server_default="Y")
    
    # Java BaseEntity: createdAt, updatedAt, deletedAt
    created_at = Column("created_at", DateTime, nullable=False, server_default=func.now())
    updated_at = Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    deleted_at = Column("deleted_at", DateTime, nullable=True)

    def __repr__(self):
        return f"<Category(id={self.id}, name={self.name})>"
