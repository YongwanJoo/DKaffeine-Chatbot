from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infrastructure.persistence.models.category_model import Category


class CategoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_by_name(self, name: str) -> Optional[Category]:
        """이름으로 카테고리 조회"""
        stmt = select(Category).where(Category.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def find_by_id(self, category_id: int) -> Optional[Category]:
        """ID로 카테고리 조회 (실제 DB 스키마에 맞게 수정)"""
        from sqlalchemy import text
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # 실제 DB 스키마 확인: category_id만 조회 (name 컬럼이 없을 수 있음)
            # 먼저 간단한 존재 여부 확인 쿼리
            logger.debug(f"🔍 [CategoryRepository] category_id={category_id} 조회 쿼리 실행")
            stmt = text("SELECT category_id FROM category WHERE category_id = :category_id AND (deleted_at IS NULL OR deleted_at = '1970-01-01') LIMIT 1")
            result = await self.session.execute(stmt, {"category_id": category_id})
            row = result.fetchone()
            
            if row:
                # Category 객체를 반환하기 위해 최소한의 정보로 생성
                category = Category()
                category.id = row[0]
                logger.debug(f"✅ [CategoryRepository] category_id={category_id} 찾음")
                return category
            else:
                logger.debug(f"⚠️ [CategoryRepository] category_id={category_id}를 찾지 못함 (DB에 존재하지 않음)")
            return None
        except Exception as e:
            # DB 스키마가 다를 수 있으므로 에러 로깅
            logger.warning(f"⚠️ [CategoryRepository] Category find_by_id failed (schema may differ): {e}")
            # 폴백: 더 간단한 쿼리 시도
            try:
                stmt = text("SELECT 1 FROM category WHERE category_id = :category_id")
                result = await self.session.execute(stmt, {"category_id": category_id})
                if result.fetchone():
                    # ID만 확인되면 Category 객체 생성
                    category = Category()
                    category.id = category_id
                    logger.debug(f"✅ [CategoryRepository] category_id={category_id} 찾음 (fallback)")
                    return category
            except Exception as e2:
                logger.warning(f"⚠️ [CategoryRepository] Category find_by_id fallback also failed: {e2}")
            return None